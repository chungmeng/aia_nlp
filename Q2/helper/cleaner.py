from helper.emoticons import EMOTICONS_EMO
from bs4 import BeautifulSoup
from html import unescape
import contractions
import demoji
import unidecode
import re

def clean_word(word):
    word = EMOTICONS_EMO.get(word, word) # Emoticon to Text
    word = contractions.fix(word) #Remove Contractions
    return word

def clean_general(text):

    text = re.sub(r'(\n|\r)', ' ', text) # Remove Newline chars
    soup = BeautifulSoup(unescape(text), 'lxml')
    text = soup.text
    text = re.sub('@\S+',' ', text) # Remove @<username>
    text = unidecode.unidecode(text) # Remove accented
    text = re.sub('(^|\s+)RT\s+',' ', text) # Remove Re-tweet symbols
    text = re.sub('http[s]?://\S+', '', text) # Remove Web Links
    text = demoji.replace_with_desc(text, sep=' ') # Emoji to Text
    text = ' '.join([clean_word(w) for w in text.split()]) # Process words
    text = re.sub(r'[^\w]', ' ', text) #Remove symbols
    text = re.sub(r'[0-9]+', '', text) #Remove numbers

    ## TO-DO Spell Checker ? ##
    return text