"""
Convert float64 and int64 columns to float32 in dataset parquet files.

This script fixes data type mismatches in parquet files by converting:
- action: list<int64> or list<float64> -> list<float32>
- observation.state: struct<state: list<int64>> or list<float64> -> struct<state: list<float32>>
- timestamp: float64 -> float32

Usage:
    python scripts/convert_float64_to_float32.py --dataset_path dataset/session_2025-10-28_15-41-34
"""

import argparse
import glob
from pathlib import Path
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_field_to_float32(field: pa.Field, field_name: str = None) -> pa.Field:
    """Recursively convert float64 and int64 fields to float32, but only for specific fields."""
    field_name = field_name or field.name
    
    # Fields to convert to float32
    convert_fields = {'action', 'timestamp', 'observation'}
    
    # Check if this field should be converted
    should_convert = any(convert_fields_name in field_name for convert_fields_name in convert_fields)
    
    if should_convert:
        if pa.types.is_float64(field.type) or pa.types.is_int64(field.type):
            return pa.field(field.name, pa.float32(), field.nullable, field.metadata)
        elif pa.types.is_list(field.type):
            element_type = field.type.value_type
            if pa.types.is_float64(element_type) or pa.types.is_int64(element_type):
                new_type = pa.list_(pa.float32())
                return pa.field(field.name, new_type, field.nullable, field.metadata)
            else:
                # Recursively handle element type if it's complex
                converted_element = convert_field_to_float32(pa.field("_elem", element_type), field_name)
                new_type = pa.list_(converted_element.type)
                return pa.field(field.name, new_type, field.nullable, field.metadata)
        elif pa.types.is_struct(field.type):
            # Struct field - recursively convert child fields
            new_fields = []
            for child_field in field.type:
                new_fields.append(convert_field_to_float32(child_field, field_name + "." + child_field.name))
            new_type = pa.struct(new_fields)
            return pa.field(field.name, new_type, field.nullable, field.metadata)
    
    return field


def convert_table(table: pa.Table) -> pa.Table:
    """Convert all float64 columns to float32 in a PyArrow table."""
    # Build target schema with float32 conversions
    new_fields = []
    for field in table.schema:
        new_fields.append(convert_field_to_float32(field))
    new_schema = pa.schema(new_fields)
    
    # Cast the entire table to the new schema
    try:
        return table.cast(new_schema)
    except Exception as e:
        logger.debug(f"Direct cast failed, attempting column-by-column conversion: {e}")
        # Fall back to column-by-column conversion
        new_columns = {}
        for i, (old_field, new_field) in enumerate(zip(table.schema, new_fields)):
            if old_field.type != new_field.type:
                logger.debug(f"Converting column '{old_field.name}': {old_field.type} -> {new_field.type}")
                try:
                    new_columns[new_field.name] = table.column(i).cast(new_field.type)
                except Exception as col_err:
                    logger.debug(f"Cast failed for {old_field.name}, trying via pylist: {col_err}")
                    # Last resort: go through Python
                    pylist = table.column(i).to_pylist()
                    new_columns[new_field.name] = pa.array(pylist, type=new_field.type)
            else:
                new_columns[new_field.name] = table.column(i)
        
        return pa.table(new_columns)


def process_file(input_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Convert a single parquet file from float64 to float32.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file (overwrites input if None)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = output_path or input_path
        
        # Read the parquet file
        logger.info(f"Reading: {input_path}")
        table = pq.read_table(input_path)
        
        # Check if conversion is needed
        has_float64 = any(
            ('float64' in str(field.type).lower() or 
             'double' in str(field.type).lower() or
             'int64' in str(field.type).lower()) and
            any(convert_name in field.name for convert_name in {'action', 'timestamp', 'observation'})
            for field in table.schema
        )
        
        if not has_float64:
            logger.info(f"  → No float64 columns found, skipping")
            return True
        
        # Convert the table
        logger.info(f"  → Converting to float32...")
        converted_table = convert_table(table)
        
        # Write back to parquet
        logger.info(f"  → Writing: {output_path}")
        pq.write_table(converted_table, str(output_path), compression='snappy')
        
        logger.info(f"  ✓ Successfully converted")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Error processing {input_path}: {e}")
        return False


def fix_index_fields(input_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Fix index fields that were incorrectly converted to float.
    Converts frame_index, episode_index, index, task_index back to int64.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file (overwrites input if None)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = output_path or input_path
        
        # Read the parquet file
        logger.info(f"Reading: {input_path}")
        table = pq.read_table(input_path)
        
        # Check if we need to fix any index fields
        index_fields = {'frame_index', 'episode_index', 'index', 'task_index'}
        needs_fix = any(
            field.name in index_fields and 
            (pa.types.is_float32(field.type) or pa.types.is_float64(field.type) or pa.types.is_float(field.type))
            for field in table.schema
        )
        
        if not needs_fix:
            logger.info(f"  → Index fields are already correct, skipping")
            return True
        
        # Fix index fields
        logger.info(f"  → Fixing index fields to int64...")
        new_columns = {}
        for i, field in enumerate(table.schema):
            col = table.column(i)
            if field.name in index_fields and (pa.types.is_float32(field.type) or pa.types.is_float64(field.type) or pa.types.is_float(field.type)):
                # Convert float to int64
                logger.debug(f"  Converting {field.name}: {field.type} -> int64")
                new_columns[field.name] = col.cast(pa.int64())
            else:
                new_columns[field.name] = col
        
        fixed_table = pa.table(new_columns)
        
        # Write back to parquet
        logger.info(f"  → Writing: {output_path}")
        pq.write_table(fixed_table, str(output_path), compression='snappy')
        
        logger.info(f"  ✓ Successfully fixed")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Error processing {input_path}: {e}")
        return False


def flatten_nested_columns(input_path: Path, output_path: Optional[Path] = None) -> bool:
    """
    Flatten nested struct columns into dot-notation columns.
    For example: observation.state (struct) -> observation.state (flat column)
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file (overwrites input if None)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = output_path or input_path
        
        # Read the parquet file
        logger.info(f"Reading: {input_path}")
        table = pq.read_table(input_path)
        
        # Check if there are nested structs
        has_nested = any(pa.types.is_struct(field.type) for field in table.schema)
        
        if not has_nested:
            logger.info(f"  → No nested structs found, skipping")
            return True
        
        # Flatten nested columns
        logger.info(f"  → Flattening nested structs...")
        new_columns = {}
        
        for i, field in enumerate(table.schema):
            col = table.column(i)
            if pa.types.is_struct(field.type):
                # Flatten struct columns
                for j, child_field in enumerate(field.type):
                    # Extract the struct field using compute API
                    child_col = pc.struct_field(col, [j])
                    # Ensure child column is explicitly float32 if it's float
                    if pa.types.is_list(child_field.type):
                        element_type = child_field.type.value_type
                        if pa.types.is_float32(element_type) or pa.types.is_float64(element_type):
                            # Cast to list of float32
                            new_type = pa.list_(pa.float32())
                            child_col = pc.cast(child_col, new_type)
                    flat_name = f"{field.name}.{child_field.name}"
                    logger.debug(f"  Flattening: {flat_name}")
                    new_columns[flat_name] = child_col
            else:
                # Non-struct columns: ensure float is float32
                if (pa.types.is_float32(field.type) or pa.types.is_float64(field.type)) and not pa.types.is_float32(field.type):
                    logger.debug(f"  Converting {field.name} to float32")
                    col = pc.cast(col, pa.float32())
                new_columns[field.name] = col
        
        flattened_table = pa.table(new_columns)
        
        # Write back to parquet
        logger.info(f"  → Writing: {output_path}")
        pq.write_table(flattened_table, str(output_path), compression='snappy')
        
        logger.info(f"  ✓ Successfully flattened")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Error processing {input_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert float64 columns to float32 in dataset parquet files'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to dataset directory (e.g., dataset/session_2025-10-28_15-41-34)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--fix-indices',
        action='store_true',
        help='Fix index fields that were incorrectly converted to float'
    )
    parser.add_argument(
        '--flatten',
        action='store_true',
        help='Flatten nested struct columns (like observation.state)'
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return 1
    
    # Find all parquet files in data/ subdirectory
    data_dir = dataset_path / 'data'
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return 1
    
    parquet_files = sorted(data_dir.glob('chunk-*/file-*.parquet'))
    
    if not parquet_files:
        logger.error(f"No parquet files found in {data_dir}")
        return 1
    
    logger.info(f"Found {len(parquet_files)} parquet files to process")
    
    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")
        for path in parquet_files:
            logger.info(f"Would process: {path}")
        return 0
    
    # Process all files
    successful = 0
    failed = 0
    
    if args.fix_indices:
        logger.info("Mode: Fix index fields")
        for i, file_path in enumerate(parquet_files, 1):
            logger.info(f"[{i}/{len(parquet_files)}]")
            if fix_index_fields(file_path):
                successful += 1
            else:
                failed += 1
    elif args.flatten:
        logger.info("Mode: Flatten nested struct columns")
        for i, file_path in enumerate(parquet_files, 1):
            logger.info(f"[{i}/{len(parquet_files)}]")
            if flatten_nested_columns(file_path):
                successful += 1
            else:
                failed += 1
    else:
        logger.info("Mode: Convert float64/int64 to float32 for action/observation/timestamp")
        for i, file_path in enumerate(parquet_files, 1):
            logger.info(f"[{i}/{len(parquet_files)}]")
            if process_file(file_path):
                successful += 1
            else:
                failed += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"  ✓ Successful: {successful}")
    logger.info(f"  ✗ Failed: {failed}")
    logger.info(f"{'='*60}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
