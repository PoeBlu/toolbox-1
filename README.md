# toolbox
Curated libraries for a faster workflow

# Phase: Data
## Annotation
- Image: [makesense.ai](https://www.makesense.ai/) 
- Text: [doccano](https://doccano.herokuapp.com/), [prodigy](https://prodi.gy/)

## Dataset
- Text: [nlp-datasets](https://github.com/niderhoff/nlp-datasets), [curse-words](https://github.com/reimertz/curse-words), [badwords](https://github.com/MauriceButler/badwords), [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words), [english-words (A text file containing over 466k English words)](https://github.com/dwyl/english-words), [10K most common words](https://github.com/first20hours/google-10000-english)
- Image: [1 million fake faces](https://archive.org/details/1mFakeFaces)
- Dataset search engine: [datasetlist](https://www.datasetlist.com/), [UCI Machine Learning Datasets](https://archive.ics.uci.edu/ml/datasets.php)

## Fetch data
- Audio: [pydub](https://github.com/jiaaro/pydub)
- Video: [pytube (download youTube vidoes)](https://github.com/nficano/pytube)
- Image: [py-image-dataset-generator (auto fetch images from web for certain search)](https://github.com/tomahim/py-image-dataset-generator)
- News: [news-please](https://github.com/fhamborg/news-please)
- PDF: [camelot](https://camelot-py.readthedocs.io/en/master/), [tabula-py](https://github.com/chezou/tabula-py)
- Remote file: [smart_open](https://github.com/RaRe-Technologies/smart_open)
- Crawling: [pyppeteer (chrome automation)](https://github.com/miyakogi/pyppeteer), [MechanicalSoup](https://github.com/MechanicalSoup/MechanicalSoup), [libextract](https://github.com/datalib/libextract)
- Google sheets: [gspread](https://github.com/burnash/gspread)
- Google drive: [gdown](https://github.com/wkentaro/gdown)
- Python API for datasets: [pydataset](https://github.com/iamaziz/PyDataset)
- Google maps location data: [geo-heatmap](https://github.com/luka1199/geo-heatmap)

## Data Augmentation
- Text: [nlpaug](https://github.com/makcedward/nlpaug)
- Image: [imgaug](https://github.com/aleju/imgaug/), [albumentations](https://github.com/albumentations-team/albumentations), [augmentor](https://github.com/mdbloice/Augmentor)

# Phase: Exploration

##  Data Preparation
- Missing values: [missingno](https://github.com/ResidentMario/missingno)
- Split images into train/validation/test: [split-folders](https://github.com/jfilter/split-folders)
- Class Imbalance: [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)

## Experiment in notebooks
- View Jupyter notebooks through CLI: [nbdime](https://github.com/jupyter/nbdime)
- Parametrize notebooks: [papermill](https://github.com/nteract/papermill)
- Access notebooks programatically: [nbformat](https://nbformat.readthedocs.io/en/latest/api.html)
- Convert notebooks to other formats: [nbconvert](https://nbconvert.readthedocs.io/en/latest/)
- Extra utilities not present in frameworks: [mlxtend](https://github.com/rasbt/mlxtend)
- Maps in notebooks: [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet)

# Phase: Feature Engineering
## Generate Features
- Automatic feature engineering: [featuretools](https://github.com/FeatureLabs/featuretools), [autopandas](https://autopandas.io/)
- Custom distance metric learning: [metric-learn](http://contrib.scikit-learn.org/metric-learn/getting_started.html)
- List of holidays: [python-holidays](https://github.com/dr-prodigy/python-holidays)

# Phase: Modeling
## Model Selection
- Bruteforce through all scikit-learn model and parameters: [auto-sklearn](https://automl.github.io/auto-sklearn), [tpot](https://github.com/EpistasisLab/tpot)
- Autogenerate ML code: [automl-gs](https://github.com/minimaxir/automl-gs), [mindsdb](https://github.com/mindsdb/mindsdb)
- Pretrained models: [modeldepot](https://modeldepot.io/browse), [pytorch-hub](https://pytorch.org/hub), [papers-with-code](https://paperswithcode.com/sota)
- Find SOTA models: [sotawhat](https://sotawhat.herokuapp.com)

## Framework extensions
- Pytorch: [Keras like summary for pytorch](https://github.com/sksq96/pytorch-summary), [skorch (wrap pytorch in scikit-learn compatible API)](https://github.com/skorch-dev/skorch)
- Einstein notation: [einops](https://github.com/arogozhnikov/einops)
- Scikit-learn: [scikit-lego](https://scikit-lego.readthedocs.io/en/latest/index.html)

## Algorithms
- Gradient Boosting: [catboost](https://catboost.ai/docs/concepts/about.html)
- Hidden Markov Models: [hmmlearn](https://github.com/hmmlearn/hmmlearn)

## NLP
- Preprocessing: [textacy](https://github.com/chartbeat-labs/textacy)
- Text Extraction from Image, Audio, PDF: [textract](https://textract.readthedocs.io/en/stable/)
- Text generation: [gp2client](https://github.com/rish-16/gpt2client), [textgenrnn](https://github.com/minimaxir/textgenrnn), [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple)
- Text summarization: [textrank](https://github.com/summanlp/textrank), [pytldr](https://github.com/jaijuneja/PyTLDR)
- Spelling correction: [JamSpell](https://github.com/bakwc/JamSpell), [pyhunspell](https://github.com/blatinier/pyhunspell), [pyspellchecker](https://github.com/barrust/pyspellchecker), [cython_hunspell](https://github.com/MSeal/cython_hunspell), [hunspell-dictionaries](https://github.com/wooorm/dictionaries), [autocorrect (can add more languages)](https://github.com/phatpiglet/autocorrect)
- Keyword extraction: [rake](https://github.com/zelandiya/RAKE-tutorial), [pke](https://github.com/boudinfl/pke)
- Multiply Choice Question Answering: [mcQA](https://github.com/mcQA-suite/mcQA)
- Sequence to sequence models: [headliner](https://github.com/as-ideas/headliner)
- Transfer learning: [finetune](https://github.com/IndicoDataSolutions/finetune)
- Translation: [googletrans](https://pypi.org/project/googletrans/)
- Embeddings: [pymagnitude (manage vector embeddings easily)](https://github.com/plasticityai/magnitude), [chakin (download pre-trained word vectors)](https://github.com/chakki-works/chakin), [sentence-transformers](https://github.com/UKPLab/sentence-transformers), [InferSent](https://github.com/facebookresearch/InferSent), [bert-as-service](https://github.com/hanxiao/bert-as-service), [sent2vec](https://github.com/NewKnowledge/nk-sent2vec)
- Multilingual support: [polyglot](https://polyglot.readthedocs.io/en/latest/index.html), [inltk (indic languages)](https://github.com/goru001/inltk), [indic_nlp](https://github.com/anoopkunchukuttan/indic_nlp_library)
- NLU: [snips-nlu](https://github.com/snipsco/snips-nlu)
- Semantic parsing: [quepy](https://github.com/machinalis/quepy)
- Inflections: [inflect](https://pypi.org/project/inflect/)
- Contractions: [pycontractions](https://pypi.org/project/pycontractions/)
- Coreference Resolution: [neuralcoref](https://github.com/huggingface/neuralcoref)
- Readability: [homer](https://github.com/wyounas/homer)
- Language Detection: [language-check](https://github.com/myint/language-check)
- Topic Modeling: [guidedlda](https://github.com/vi3k6i5/guidedlda)
- Clustering: [spherecluster (kmeans with cosine distance)](https://github.com/jasonlaska/spherecluster), [kneed (automatically find number of clusters from elbow curve)](https://github.com/arvkevi/kneed)
- Metrics: [seqeval (NER, POS tagging)](https://github.com/chakki-works/seqeval)
- String match: [jellyfish (perform string and phonetic comparison)](https://pypi.org/project/jellyfish/),[flashtext (superfast extract and replace keywords)](https://github.com/vi3k6i5/flashtext), [pythonverbalexpressions: (verbally describe regex)](https://github.com/VerbalExpressions/PythonVerbalExpressions), [commonregex (readymade regex for email/phone etc)](https://github.com/madisonmay/CommonRegex)
- Sentiment: [vaderSentiment (rule based)](https://github.com/cjhutto/vaderSentiment)
- Text distances: [textdistance](https://github.com/life4/textdistance), [editdistance](https://github.com/aflc/editdistance)
- PID removal: [scrubadub](https://scrubadub.readthedocs.io/en/stable/#)
- Profanity detection: [profanity-check](https://github.com/vzhou842/profanity-check)
- wordclouds: [stylecloud](https://github.com/minimaxir/stylecloud)

## Speech Recognition
- Library: [speech_recognition](https://github.com/Uberi/speech_recognition)

## RecSys
- Factorization machines (FM), and field-aware factorization machines (FFM): [xlearn](https://github.com/aksnzhy/xlearn)
- Scikit-learn like API: [surprise](https://github.com/NicolasHug/Surprise)

## Computer Vision
- Image processing: [scikit-image](https://github.com/scikit-image/scikit-image), [imutils](https://github.com/jrosebr1/imutils)
- Segmentation Models in Keras: [segmentation_models](https://github.com/qubvel/segmentation_models)
- Face recognition: [face_recognition](https://github.com/ageitgey/face_recognition), [face-alignment (find facial landmarks)](https://github.com/1adrianb/face-alignment)
- Face swapping: [faceit](https://github.com/goberoi/faceit)
- Video summarization: [videodigest](https://github.com/agermanidis/videodigest)
- Semantic search over videos: [scoper](https://github.com/RameshAditya/scoper)
- OCR: [keras-ocr](https://github.com/faustomorales/keras-ocr), [pytesseract](https://github.com/madmaze/pytesseract)

## Timeseries
- Predict Time Series: [prophet](https://facebook.github.io/prophet/docs/quick_start.html#python-api)

# Phase: Monitoring
## Monitor training process
- Learning curve: [lrcurve (plot realtime learning curve in Keras)](https://github.com/AndreasMadsen/python-lrcurve), [livelossplot](https://github.com/stared/livelossplot)
- Notifications: [knockknock (get notified by slack/email)](https://github.com/huggingface/knockknock), [jupyter-notify (notify when task is completed in jupyter)](https://github.com/ShopRunner/jupyter-notify)

# Phase: Optimization
## Hyperparameter Optimization
- Keras: [keras-tuner](https://github.com/keras-team/keras-tuner)
- General: [hyperopt](https://github.com/hyperopt/hyperopt)

## Interpretability
- Visualize keras models: [keras-vis](https://github.com/raghakot/keras-vis)
- Interpret models: [eli5](https://eli5.readthedocs.io/en/latest/), [lime](https://github.com/marcotcr/lime), [shap](https://github.com/slundberg/shap), [alibi](https://github.com/SeldonIO/alibi), [tf-explain](https://github.com/sicara/tf-explain), [treeinterpreter](https://github.com/andosa/treeinterpreter)
- Interpret BERT: [exbert](http://exbert.net/exBERT.html?sentence=I%20liked%20the%20music&layer=0&heads=..0,1,2,3,4,5,6,7,8,9,10,11&threshold=0.7&tokenInd=null&tokenSide=null&maskInds=..9&metaMatch=pos&metaMax=pos&displayInspector=null&offsetIdxs=..-1,0,1&hideClsSep=true)
- Interpret word2vec: [word2viz](https://lamyiowce.github.io/word2viz/)

## Visualization
- Draw CNN figures: [nn-svg](http://alexlenail.me/NN-SVG/LeNet.html)
- Visualization for scikit-learn: [yellowbrick](https://www.scikit-yb.org/en/latest/index.html), [scikit-plot](https://scikit-plot.readthedocs.io/en/stable/metrics.html)
- XKCD like charts: [chart.xkcd](https://timqian.com/chart.xkcd/)
- Convert matplotlib charts to D3 charts: [mpld3](http://mpld3.github.io/index.html)
- Generate graphs using markdown: [mermaid](https://mermaid-js.github.io/mermaid/#/README)
- Visualize topics models: [pyldavis](https://pyldavis.readthedocs.io/en/latest/)

# Phase: Production
## Scalability
- Parallelize .apply in Pandas: [pandarallel](https://github.com/nalepae/pandarallel), [swifter](https://github.com/jmcarpenter2/swifter)

## Bechmarking
- Profile pytorch layers: [torchprof](https://github.com/awwong1/torchprof)

## API
- Read config files: [config](https://pypi.org/project/config/), [python-decouple](https://github.com/henriquebastos/python-decouple)
- Data Validation: [schema](https://github.com/keleshev/schema), [jsonschema](https://pypi.org/project/jsonschema/), [cerebrus](https://github.com/pyeve/cerberus), [pydantic](https://pydantic-docs.helpmanual.io/), [marshmallow](https://marshmallow.readthedocs.io/en/stable/)
- Enable CORS in Flask: [flask-cors](https://flask-cors.readthedocs.io/en/latest/)
- Cache results of functions: [cachetools](https://pypi.org/project/cachetools/)
- Authentication: [pyjwt (JWT)](https://github.com/jpadilla/pyjwt)
- Task Queue: [rq](https://github.com/rq/rq)

## Deployment
- Transpiling: [sklearn-porter (transpile sklearn model to C, Java, JavaScript and others)](https://github.com/nok/sklearn-porter), [m2cgen](https://github.com/BayesWitnesses/m2cgen)

## Dashboards
- Generate frontend with python: [streamlit](https://github.com/streamlit/streamlit)

## Adversarial testing
- Generate images to fool model: [foolbox](https://github.com/bethgelab/foolbox)
- Generate phrases to fool NLP models: [triggers](https://www.ericswallace.com/triggers)

## Python libraries
- Datetime compatible API for Bikram Sambat: [nepali-date](https://github.com/arneec/nepali-date)
- bloom filter: [python-bloomfilter](https://github.com/jaybaird/python-bloomfilter)
- Run python libraries in sandbox: [pipx](https://github.com/pipxproject/pipx)
- Pretty print tables in CLI: [tabulate](https://pypi.org/project/tabulate/)
- Leaflet maps from python: [folium](https://python-visualization.github.io/folium/)
- Debugging: [PySnooper](https://github.com/cool-RR/PySnooper)
- Pickling extended: [cloudpickle](https://github.com/cloudpipe/cloudpickle)
