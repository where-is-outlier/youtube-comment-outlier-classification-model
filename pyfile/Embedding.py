import torch

# 임베딩 함수 정의
def embed_text(text_series, tokenizer, model):
    embeddings = []
    tokens = tokenizer(
        text_series.tolist(),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=5,
    )
    tokens = {key: value.to("cuda") for key, value in tokens.items()}  # GPU로 이동
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()  # 결과를 CPU로 이동
    embeddings.append(embedding)
    return embeddings