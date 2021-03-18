from datetime import timedelta
from urllib.request import urlopen
from xml.dom.minidom import parseString
import pandas as pd
from IPython import embed


if __name__ == '__main__':

    df = pd.read_pickle('continguts_indexacio.pkl')
    df['url'] = df.apply(lambda row: row['mp4_500'] if not isinstance(row['mp4_500'], float) else row['mp4_500_es'], axis=1)# df['url'].tolist()

    sample = df[df['content_id'] == 5745104]
    url = sample['url'].values[0]
    embed()

