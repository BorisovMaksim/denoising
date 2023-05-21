import os
import wandb
from huggingface_hub import HfApi
from pathlib import Path
import huggingface_hub
import ssl
import os
os.environ['CURL_CA_BUNDLE'] = ''

ssl._create_default_https_context = ssl._create_unverified_context

class Uploader:
    def __init__(self, entity, project, run_name, repo_id, username):
        self.entity = entity
        self.project = project
        self.run_name = run_name
        self.hf_api = HfApi()
        self.wandb_api = wandb.Api()
        self.repo_id = repo_id
        self.username = username
        huggingface_hub.login(os.environ.get('HUGGINGFACE_TOKEN'))

    def get_model_from_wandb_run(self):
        runs = self.wandb_api.runs(f"{self.entity}/{self.project}",
                        # order='+summary_metrics.train_pesq'
                        )
        run = [run for run in runs if run.name == self.run_name][0]
        artifacts = run.logged_artifacts()
        best_model = [artifact for artifact in artifacts if artifact.type == 'model'][0]
        artifact_dir = best_model.download()
        model_path = list(Path(artifact_dir).glob("*.pt"))[0].absolute().as_posix()
        print(f"Model validation score = {best_model.metadata['Validation score']}")
        return model_path

    def upload_to_HF(self):
        model_path = self.get_model_from_wandb_run()
        self.hf_api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=Path(model_path).name,
            repo_id=f'{self.username}/{self.repo_id}',
        )

    def create_repo(self):
        self.hf_api.create_repo(repo_id=self.repo_id, exist_ok=True)



if __name__ == '__main__':
    uploader = Uploader(entity='borisovmaksim',
                        project='denoising',
                        run_name='wav_normalization',
                        repo_id='demucs',
                        username='BorisovMaksim')
    uploader.create_repo()
    uploader.upload_to_HF()

