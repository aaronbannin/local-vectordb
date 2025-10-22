import cohere

model_name = "embed-v4.0"
api_key = "pa6sRhnVAedMVClPAwoCvC1MjHKEwjtcGSTjWRMd"
# pa6sRhnVAedMVClPAwoCvC1MjHKEwjtcGSTjWRMd
# rQsWxQJOK89Gp87QHo6qnGtPiWerGJOxvdg59o5f
input_type_embed = "search_document"

co = cohere.Client(api_key)


def get_embeddings_bulk(texts: list[str]) -> list[list[float]]:
    response = co.embed(texts=texts, model=model_name, input_type=input_type_embed)
    embeds: list[list[float]] = response.embeddings

    return embeds
