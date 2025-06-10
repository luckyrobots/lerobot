from huggingface_hub import create_tag, delete_tag, HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

# Replace with your dataset repo id and desired tag
repo_id = "luckyrobots/so100_Dataset_250_V0.1"
tag = "v2.1"  # or any version string you want

# Check if the repo exists before proceeding
def repo_exists(repo_id):
    api = HfApi()
    try:
        api.repo_info(repo_id, repo_type="dataset")
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as e:
        print(f"Unexpected error while checking repo existence: {e}")
        return False

if not repo_exists(repo_id):
    print(f"ERROR: The dataset repo '{repo_id}' does not exist on Hugging Face. Please create it first.")
    exit(1)

# Attempt to delete the tag if it exists, to allow overwriting
try:
    print(f"Attempting to delete tag '{tag}' from repo '{repo_id}' if it exists...")
    delete_tag(repo_id, tag=tag, repo_type="dataset")
    print(f"Successfully deleted tag '{tag}' or it did not exist.")
except HfHubHTTPError as e:
    # If the error is that the tag doesn't exist (404), it's fine.
    # If it's another error, we might want to raise it or log it.
    if e.response.status_code == 404:
        print(f"Tag '{tag}' did not exist, no need to delete.")
    else:
        print(f"Could not delete tag '{tag}' due to an unexpected error: {e}")
        # Optionally, re-raise the error if it's critical
        # raise
except Exception as e:
    print(f"An unexpected error occurred during tag deletion: {e}")
    # Optionally, re-raise the error
    # raise

# Create the tag on the dataset repo
print(f"Creating tag '{tag}' on repo '{repo_id}'...")
create_tag(repo_id, tag=tag, repo_type="dataset")
print(f"Successfully created tag '{tag}' on repo '{repo_id}'.")

# --- Change Log ---
# 2024-06-02: Modified by AI assistant.
# - Added a check to ensure the repo exists before attempting tag operations.
# - Prints a clear error and exits if the repo does not exist.
# --- End Change Log ---