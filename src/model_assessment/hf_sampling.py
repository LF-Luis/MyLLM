import logging
from dataclasses import fields

from huggingface_hub import hf_hub_download

from src.model import LLM
from src.params import TParams
from src.utils.handle_ddp import DDPHandler
from src.utils.root import create_temp_data_dir
from src.model_assessment.sampling import sample
from src.model_configs.my_llm_config import get_llm_config
from src.model_utils.checkpoint_utils import load_checkpoint


log = logging.getLogger(__name__)
HF_MODEL_DIR = "hf_model"
REPO_ID = "LF-Luis/LF_LLM-269M"
MODEL_NAME = "pytorch_model.pth"


def download_model():
    '''
    Download model from Hugging Face.
    '''
    model_dir = create_temp_data_dir(HF_MODEL_DIR)
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_NAME, cache_dir=model_dir)
    log.info(f"Model downloaded to: {model_path}")
    return model_path

def get_hf_model(ddp):
    model_path = download_model()

    hParams, _ = get_llm_config()
    model = LLM(hParams)
    load_checkpoint(
        model,
        optimizer=None,
        scheduler=None,
        filepath=model_path,
        load_random_state=False
    )
    model.to(ddp.assigned_device)
    return model

def hf_sample(model, ddp, tParams, prompt):
    sample(model, ddp, prompt, tParams)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tParams = TParams(**{f.name: None for f in fields(TParams)})
    tParams.sampling_tokens = 5
    tParams.sampling_batch = 2
    tParams.sampling_top_k = 50

    ddp = DDPHandler()
    model = get_hf_model(ddp)
    hf_sample(model, ddp, tParams, "Let's talk about Rome, Rome is")
