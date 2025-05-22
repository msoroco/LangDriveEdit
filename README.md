# LangDriveEdit


# Code organization

| Branch              | Purpose                                      |
|---------------------|----------------------------------------------|
| `main`              | The chat_GPT prompts and mask creation for real-world data     |
| `realworld_metadata`| The Synthetic data creation       |
| `training_ue`| Training code for Ultra Edit experiments          |
| `training_ip2p`               | Training code for Instruct Pix 2 Pix experiments      |

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
## download realistic train set
```
aws s3 cp s3://lang-drive-edit/boreas_train boreas_train --recursive --no-sign-request
```

## download realistic test set
```
aws s3 cp s3://lang-drive-edit/boreas_test boreas_test --recursive --no-sign-request
```
