from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

def dataframe_info(df):
    print(f'INFO : DataFrame has {df.shape[0]} rows and {df.shape[1]} columns')
    print(f'INFO : Column Names are {df.columns}')

def check_mapping(df):
    label_dict={}
    for n in df['label'].unique():
        topic = df[df['label']==n]['topic'].unique()
        if not len(topic)==1:
            raise Exception(f'ERROR : Label to Topic ({n}=>{topic}) Mapping not 1:1')
        print(f'INFO : {n}=>{topic}')
        label_dict[topic[0]]=n
    return label_dict

def peek_tfidf(text_series, top=12, ngram_range=(3,3)):
    count_vectorizer = CountVectorizer(ngram_range = ngram_range)
    X1 = count_vectorizer.fit_transform(text_series) 
    features = (count_vectorizer.get_feature_names())

    # Applying TFIDF
    tfidf_vectorizer = TfidfVectorizer(ngram_range = ngram_range)
    X2 = tfidf_vectorizer.fit_transform(text_series)
    scores = (X2.toarray())

    # Getting top ranking features
    sums = X2.sum(axis = 0)
    data1 = []
    for col, term in enumerate(features):
        data1.append( (term, sums[0,col] ))
    df_rank = pd.DataFrame(data1, columns = ['term','rank'])
    return df_rank.sort_values('rank', ascending = False)[:top]

def plot_wordcloud(text, savefig_path=''):
    exclude_words=set(['coronavirus', 'mers', 'virus', 'ebola', 'disease', 'covid', 'corona'])
    wcloud=WordCloud(stopwords=STOPWORDS | exclude_words)
    wcloud.generate(text)
    plt.figure(figsize=(20,20))
    plt.imshow(wcloud, interpolation='bilinear')
    plt.axis("off")
    
    if savefig_path:
        try:
            plt.savefig(savefig_path, dpi=300, bbox_inches = "tight")
            print('INFO : Saved Image of Confusion Matrix at ', savefig_path)
        except:
            print('WARNING : Failed to Save Image to ', savefig_path)
    
    plt.show()