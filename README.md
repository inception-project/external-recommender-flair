# external-recommender-flair

This project contains an example external recommender for the [INCEpTION annotation platform](https://inception-project.github.io). It is used to recommend possible name entities and part-of-speech tags to an annotator in order to speed up annotation and improve annotation quality. It uses [Zalando Flair](https://github.com/zalandoresearch/flair) internally to do this predictions and [Flask](http://flask.pocoo.org) as the web framework.

## Installation

This project uses Python ≥ 3.6, `Flask`, `Zalando Flair` and `dkpro-cassis`. It is recommended to install these dependencies in a vitural environment:

    virtualenv venv --python=python3 --no-site-packages
    source venv/bin/activate
    python -m pip install git+https://github.com/dkpro/dkpro-cassis
    pip install flask
    pip install flair

When the recommender is deployed on a Gunicorn server:

    pip install gunicorn

## Usage

After everything has been set up, the recommender then can be started from the command line by calling

    python app_flair.py --pos ${POS_MODEL_NAME} --ner ${NER_MODEL_NAME} --sentiment ${SENTIMENT_CLASSIFIER_MODEL_NAME}
    
where ${POS_MODEL_NAME} is the name of the POS-tagging model, ${NER_MODEL_NAME} is the name of NER model and ${SENTIMENT_CLASSIFIER_MODEL_NAME} is the name of the text classification model. A list of pretrained models can be found on the [flair page](https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md).

When used in production, the recommender can be deployed on an actual application server Gunicorn by calling:

    gunicorn app_flair:app -c gunicorn.conf

where gunicorn.conf is the config file. The following line of the config file is used to configure the models in Gunicorn:

    raw_env = ["pos_model=${POS_MODEL_NAME}","ner_model=${NER_MODEL_NAME}","sentiment_model=${SENTIMENT_CLASSIFIER_MODEL_NAME}"]

More information about the configuration of Gunicorn can be found on the [Gunicorn page](http://docs.gunicorn.org/en/stable/settings.html#config-file).

For now, the recommender supports the following models:
 
 Name  | Task  | Training Dataset
 ---- | ----- | ------  
 'ner'  | 4-class Named Entity Recognition | Conll-03
 'ner-fast'  | 4-class Named Entity Recognition(smaller model) | Conll-03
 'ner-ontonotes'  | 18-class Named Entity Recognition | Ontonotes
 'ner-ontonotes-fast'  | 18-class Named Entity Recognition(smaller model) | Ontonotes
 'pos' | Part-of-Speech Tagging | Ontonotes
 'pos-fast' | Part-of-Speech Tagging(smaller model) | Ontonotes
 'en-sentiment' | detecting positive and negative sentiment | movie reviews from [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/)
 
 In order to simplify the setup, the recommender is also packaged as a docker file. The installation of Docker in different platforms can be found on [Docker page](https://docs.docker.com/install/). With the given Dockerfile you can build a container image by running:

    docker build -t <name_of_container> .

and test your docker image locally by running( add "-d" option if you want to run the container in the background ):
    
    docker run -it -p 5000:5000 <name_of_container>

More details about Docker can be found on [Docker Documentation](https://docs.docker.com).

### Tagset

`flair_ner.json` contains 4-class name entity tags from Conll-03, `flair_ner_ontonotes.json` contains 12-class name entity tags from Ontonotes and `flair_pos.json` contains pos-of-speech tags from Ontonotes. When you annotate your text with Flair external recommender, you need first import the tagset files into your project in INCEpTION as "Settings -> Tagsets -> Choose Files -> Format : JSON -> Submit".

## Development

The tests can be run via

    python -m unittest discover
