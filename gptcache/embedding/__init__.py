__all__ = [
    "OpenAI",
    "OpenAI3S",
    "OpenAI3L",
    "Huggingface",
    "SBERT",
    "Cohere",
    "Onnx",
    "FastText",
    "Data2VecAudio",
    "Timm",
    "ViT",
    "LangChain",
    "Rwkv",
    "PaddleNLP",
    "UForm",
]


from gptcache.utils.lazy_import import LazyImport

openai = LazyImport("openai", globals(), "gptcache.embedding.openai")
openai3s = LazyImport("openai3s", globals(), "gptcache.embedding.openai3s")
openai3l = LazyImport("openai3l", globals(), "gptcache.embedding.openai3l")
huggingface = LazyImport("huggingface", globals(), "gptcache.embedding.huggingface")
sbert = LazyImport("sbert", globals(), "gptcache.embedding.sbert")
onnx = LazyImport("onnx", globals(), "gptcache.embedding.onnx")
cohere = LazyImport("cohere", globals(), "gptcache.embedding.cohere")
fasttext = LazyImport("fasttext", globals(), "gptcache.embedding.fasttext")
data2vec = LazyImport("data2vec", globals(), "gptcache.embedding.data2vec")
timm = LazyImport("timm", globals(), "gptcache.embedding.timm")
vit = LazyImport("vit", globals(), "gptcache.embedding.vit")
langchain = LazyImport("langchain", globals(), "gptcache.embedding.langchain")
rwkv = LazyImport("rwkv", globals(), "gptcache.embedding.rwkv")
paddlenlp = LazyImport("paddlenlp", globals(), "gptcache.embedding.paddlenlp")
uform = LazyImport("uform", globals(), "gptcache.embedding.uform")


def Cohere(model="large", api_key=None):
    return cohere.Cohere(model, api_key)


def OpenAI(model="text-embedding-ada-002", api_key=None):
    return openai.OpenAI(model, api_key)


def OpenAI3S(model="text-embedding-3-small", api_key=None):
    return openai3s.OpenAI3S(model, api_key)


def OpenAI3L(model="text-embedding-3-large", api_key=None):
    return openai3l.OpenAI3L(model, api_key)


def Huggingface(model="distilbert-base-uncased"):
    return huggingface.Huggingface(model)


def SBERT(model="all-MiniLM-L6-v2"):
    return sbert.SBERT(model)


def Onnx(model="GPTCache/paraphrase-albert-onnx"):
    return onnx.Onnx(model)


def FastText(model="en", dim=None):
    return fasttext.FastText(model, dim)


def Data2VecAudio(model="facebook/data2vec-audio-base-960h"):
    return data2vec.Data2VecAudio(model)


def Timm(model="resnet50", device="default"):
    return timm.Timm(model, device)


def ViT(model="google/vit-base-patch16-384"):
    return vit.ViT(model)


def LangChain(embeddings, dimension=0):
    return langchain.LangChain(embeddings, dimension)


def Rwkv(model="sgugger/rwkv-430M-pile"):
    return rwkv.Rwkv(model)


def PaddleNLP(model="ernie-3.0-medium-zh"):
    return paddlenlp.PaddleNLP(model)


def UForm(model="unum-cloud/uform-vl-multilingual", embedding_type="text"):
    return uform.UForm(model, embedding_type)
