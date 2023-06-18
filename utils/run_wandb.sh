docker pull wandb/local:latest
docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local:latest
