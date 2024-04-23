from fastapi import FastAPI, HTTPException
from textdistance import (hamming, mlipns, levenshtein, damerau_levenshtein, jaro_winkler, strcmp95, needleman_wunsch,
                          gotoh, smith_waterman,
                          jaccard, sorensen, tversky, overlap, tanimoto, cosine,monge_elkan, bag,
                          lcsseq, lcsstr, ratcliff_obershelp,
                          arith_ncd, rle_ncd, bwtrle_ncd,
                          sqrt_ncd, entropy_ncd,
                          bz2_ncd, lzma_ncd, zlib_ncd,
                          mra, editex,
                          prefix, postfix, length, identity, matrix)

from app.api.api import api_router
from app.api.heartbeat import heartbeat_router
from app.core.config import settings
from app.core.event_handler import start_app_handler, stop_app_handler
from fastapi.staticfiles import StaticFiles  # для публикации index.html

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(heartbeat_router)
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["ML API"])

app.add_event_handler("startup", start_app_handler(app, settings.MODEL_PATH))
app.add_event_handler("shutdown", stop_app_handler(app))

# =========================================================
# Монтирование статических файлов (например, HTML, CSS, JS)
# Добавлено, чтобы опубликовать на localhost страницы index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
# ==========================================================


# REQUESTS
# ==========================================================
# ==========================================================

def calculate_method(method, line1, line2):
    # Edit based
    if method == 'hamming':
        similarity = hamming.normalized_similarity(line1, line2)
        return similarity
    elif method == 'mlipns':
        similarity = mlipns.normalized_similarity(line1, line2)
        return similarity
    elif method == 'levenshtein':
        similarity = levenshtein.normalized_similarity(line1, line2)
        return similarity
    elif method == 'damerau_levenshtein':
        similarity = damerau_levenshtein.normalized_similarity(line1, line2)
        return similarity
    elif method == 'jaro_winkler':
        similarity = jaro_winkler.normalized_similarity(line1, line2)
        return similarity
    elif method == 'strcmp95':
        similarity = strcmp95.normalized_similarity(line1, line2)
        return similarity
    elif method == 'needleman_wunsch':
        similarity = needleman_wunsch.normalized_similarity(line1, line2)
        return similarity
    elif method == 'gotoh':
        similarity = gotoh.normalized_similarity(line1, line2)
        return similarity
    elif method == 'smith_waterman':
        similarity = smith_waterman.normalized_similarity(line1, line2)
        return similarity
    # Token based
    elif method == 'jaccard':
        similarity = jaccard.normalized_similarity(line1, line2)
        return similarity
    elif method == 'sorensen':
        similarity = sorensen.normalized_similarity(line1, line2)
        return similarity
    elif method == 'tversky':
        similarity = tversky.normalized_similarity(line1, line2)
        return similarity
    elif method == 'overlap':
        similarity = overlap.normalized_similarity(line1, line2)
        return similarity
    elif method == 'tanimoto':
        similarity = tanimoto.normalized_similarity(line1, line2)
        return similarity
    elif method == 'cosine':
        similarity = cosine.normalized_similarity(line1, line2)
        return similarity
    elif method == 'tversky':
        similarity = tversky.normalized_similarity(line1, line2)
        return similarity
    elif method == 'monge_elkan':
        similarity = monge_elkan.normalized_similarity(line1, line2)
        return similarity
    elif method == 'bag':
        similarity = bag.normalized_similarity(line1, line2)
        return similarity
    # Sequence based
    elif method == 'lcsseq':
        similarity = lcsseq.normalized_similarity(line1, line2)
        return similarity
    elif method == 'lcsstr':
        similarity = lcsstr.normalized_similarity(line1, line2)
        return similarity
    elif method == 'ratcliff_obershelp':
        similarity = ratcliff_obershelp.normalized_similarity(line1, line2)
        return similarity
    # Classic compression algorithms
    elif method == 'arith_ncd':
        similarity = arith_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'rle_ncd':
        similarity = rle_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'bwtrle_ncd':
        similarity = bwtrle_ncd.normalized_similarity(line1, line2)
        return similarity
    # Normal compression algorithms
    elif method == 'sqrt_ncd':
        similarity = sqrt_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'entropy_ncd':
        similarity = entropy_ncd.normalized_similarity(line1, line2)
        return similarity
    # Work in progress algorithms that compare two strings as array of bits
    elif method == 'bz2_ncd':
        similarity = bz2_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'lzma_ncd':
        similarity = lzma_ncd.normalized_similarity(line1, line2)
        return similarity
    elif method == 'zlib_ncd':
        similarity = zlib_ncd.normalized_similarity(line1, line2)
        return similarity
    # Phonetic
    elif method == 'mra':
        similarity = mra.normalized_similarity(line1, line2)
        return similarity
    elif method == 'editex':
        similarity = editex.normalized_similarity(line1, line2)
        return similarity
    # Simple
    elif method == 'prefix':
        similarity = prefix.normalized_similarity(line1, line2)
        return similarity
    elif method == 'postfix':
        similarity = postfix.normalized_similarity(line1, line2)
        return similarity
    elif method == 'length':
        similarity = length.normalized_similarity(line1, line2)
        return similarity
    elif method == 'identity':
        similarity = identity.normalized_similarity(line1, line2)
        return similarity
    elif method == 'matrix':
        similarity = matrix.normalized_similarity(line1, line2)
        return similarity
    else:
        return None

def calculate_similarity(data):
    method = data.get('method')
    line1 = data.get('line1')
    line2 = data.get('line2')
    return calculate_method(method, line1, line2)


@app.post("/calculate_similarity/")
async def calculate_similarity_endpoint(data: dict):
    similarity = calculate_similarity(data)
    if similarity is None:
        raise HTTPException(status_code=400, detail="Invalid method specified")

    response_data = {
        'method': data['method'],
        'line1': data['line1'],
        'line2': data['line2'],
        'similarity': similarity
    }
    return response_data

# ==========================================================
# ==========================================================

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")


