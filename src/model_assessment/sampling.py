import logging

import torch
import tiktoken


log = logging.getLogger(__name__)
TOKENIZER = tiktoken.get_encoding("r50k_base")


def sample(model, ddp, prompt, tParams):
    '''
    Sample single 'prompt' multiple times.
    '''
    model.eval()
    _sample(model, ddp, prompt, tParams)
    model.train()


def multi_sample(model, ddp, prompts, tParams):
    '''
    Sample multiple 'prompts' multiple times.
    '''
    model.eval()
    prompt_shard = [val for val in prompts[ddp.local_rank::ddp.world_size]]
    for prompt in prompt_shard:
        _sample(model, ddp, prompt, tParams)
    model.train()
    

def _sample(model, ddp, prompt, tParams):
    '''
    Sample 'model' to text-complete 'prompt'. This will be done 'batch_size' number
    of times. Not going to search for EOT token, just stop at 'sampling_tokens' number of tokens.

    Using top-k sampling.

    #TODO: Add temperature.
    #TODO: Add top-p sampling following top-k.
    '''

    sampling_tokens = tParams.sampling_tokens
    batch_size = tParams.sampling_batch
    top_k = tParams.sampling_top_k

    input_ids = TOKENIZER.encode(prompt)
    sequence = torch.tensor([input_ids] * batch_size, device=ddp.assigned_device)

    with torch.no_grad():
        for _ in range(sampling_tokens):
            with torch.autocast(device_type=ddp.device_type, dtype=torch.bfloat16):
                last_tokens_logits = model(sequence)[0][:, -1, :]

            # Sample only from top-k probabilities
            top_k_logits, top_k_indices = torch.topk(last_tokens_logits, top_k, dim=-1)
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
            sampled_indices = torch.multinomial(probs, num_samples=1)

            next_tokens = top_k_indices.gather(-1, sampled_indices)
            sequence = torch.cat([sequence, next_tokens], dim=-1)

    # Decode and log generated text
    for output in sequence:
        decoded_text = TOKENIZER.decode(output.tolist())
        log.info(decoded_text)
