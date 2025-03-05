
from typing import Optional, Union, Dict, List

import warnings
import os
import logging
logger = logging.get_logger(__name__)

class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def _create_repo(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        """
        Create the repo if needed, cleans up repo_id with deprecated kwargs `repo_url` and `organization`, retrieves
        the token.
        """
        if repo_url is not None:
            warnings.warn(
                "The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` "
                "instead."
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_url` are both specified. Please set only the argument `repo_id`."
                )
            repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")
        if organization is not None:
            warnings.warn(
                "The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your "
                "organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`)."
            )
            if not repo_id.startswith(organization):
                if "/" in repo_id:
                    repo_id = repo_id.split("/")[-1]
                repo_id = f"{organization}/{repo_id}"

        # url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        # return url.repo_id
        return None

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        files_timestamps: Dict[str, float],
        commit_message: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        revision: str = None,
        commit_description: str = None,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        # if commit_message is None:
        #     if "Model" in self.__class__.__name__:
        #         commit_message = "Upload model"
        #     elif "Config" in self.__class__.__name__:
        #         commit_message = "Upload config"
        #     elif "Tokenizer" in self.__class__.__name__:
        #         commit_message = "Upload tokenizer"
        #     elif "FeatureExtractor" in self.__class__.__name__:
        #         commit_message = "Upload feature extractor"
        #     elif "Processor" in self.__class__.__name__:
        #         commit_message = "Upload processor"
        #     else:
        #         commit_message = f"Upload {self.__class__.__name__}"
        # modified_files = [
        #     f
        #     for f in os.listdir(working_dir)
        #     if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        # ]

        # # filter for actual files + folders at the root level
        # modified_files = [
        #     f
        #     for f in modified_files
        #     if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))
        # ]

        # operations = []
        # # upload standalone files
        # for file in modified_files:
        #     if os.path.isdir(os.path.join(working_dir, file)):
        #         # go over individual files of folder
        #         for f in os.listdir(os.path.join(working_dir, file)):
        #             operations.append(
        #                 CommitOperationAdd(
        #                     path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f)
        #                 )
        #             )
        #     else:
        #         operations.append(
        #             CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file)
        #         )

        # if revision is not None:
        #     create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)

        # logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        # return create_commit(
        #     repo_id=repo_id,
        #     operations=operations,
        #     commit_message=commit_message,
        #     commit_description=commit_description,
        #     token=token,
        #     create_pr=create_pr,
        #     revision=revision,
        # )

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = True,
        revision: str = None,
        commit_description: str = None,
        tags: Optional[List[str]] = None,
        **deprecated_kwargs,
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
                Google Colab instances without any CPU OOM issues.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights in safetensors format for safer serialization.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            tags (`List[str]`, *optional*):
                List of tags to push on the Hub.

        Examples:

        ```python
        from transformers import {object_class}

        {object} = {object_class}.from_pretrained("google-bert/bert-base-cased")

        # Push the {object} to your namespace with the name "my-finetuned-bert".
        {object}.push_to_hub("my-finetuned-bert")

        # Push the {object} to an organization with the name "my-finetuned-bert".
        {object}.push_to_hub("huggingface/my-finetuned-bert")
        ```
        """
        use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
        ignore_metadata_errors = deprecated_kwargs.pop("ignore_metadata_errors", False)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        repo_path_or_name = deprecated_kwargs.pop("repo_path_or_name", None)
        if repo_path_or_name is not None:
            # Should use `repo_id` instead of `repo_path_or_name`. When using `repo_path_or_name`, we try to infer
            # repo_id from the folder path, if it exists.
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead.",
                FutureWarning,
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_path_or_name` are both specified. Please set only the argument `repo_id`."
                )
            if os.path.isdir(repo_path_or_name):
                # repo_path: infer repo_id from the path
                repo_id = repo_id.split(os.path.sep)[-1]
                working_dir = repo_id
            else:
                # repo_name: use it as repo_id
                repo_id = repo_path_or_name
                working_dir = repo_id.split("/")[-1]
        else:
            # Repo_id is passed correctly: infer working_dir from it
            working_dir = repo_id.split("/")[-1]

        # Deprecation warning will be sent after for repo_url and organization
        repo_url = deprecated_kwargs.pop("repo_url", None)
        organization = deprecated_kwargs.pop("organization", None)

        # repo_id = self._create_repo(
        #     repo_id, private=private, token=token, repo_url=repo_url, organization=organization
        # )

        # # Create a new empty model card and eventually tag it
        # model_card = create_and_tag_model_card(
        #     repo_id, tags, token=token, ignore_metadata_errors=ignore_metadata_errors
        # )

        # if use_temp_dir is None:
        #     use_temp_dir = not os.path.isdir(working_dir)

        # with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
        #     files_timestamps = self._get_files_timestamps(work_dir)

        #     # Save all files.
        #     self.save_pretrained(work_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)

        #     # Update model card if needed:
        #     model_card.save(os.path.join(work_dir, "README.md"))

        #     return self._upload_modified_files(
        #         work_dir,
        #         repo_id,
        #         files_timestamps,
        #         commit_message=commit_message,
        #         token=token,
        #         create_pr=create_pr,
        #         revision=revision,
        #         commit_description=commit_description,
        #     )
