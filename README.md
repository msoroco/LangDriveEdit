# LangDriveEdit


# Code organization

The Synthetic data creation is in the `new_metadata` branch.

The chat_GPT prompts and mask creation for real-world data is in the `main` branch
---

# download cmds


## require installation of aws cli:

## download synthetic train set
```
aws s3 cp s3://lang-drive-edit/synthetic_train synthetic_train --recursive --no-sign-request
```
## download synthetic test set
```
aws s3 cp s3://lang-drive-edit/synthetic_test synthetic_test --recursive --no-sign-request
```
