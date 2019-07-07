import urllib.request as request
import json
import pickle
from datetime import datetime, timedelta

DOWNLOAD_METADATA_PATH = r"C:\TFM\videos\download\videos_metadata.pickle"

def get_all_video_in_channel(channel_id,init_date,end_date):
    api_key = "AIzaSyDOpbtVHeHNveSl8jXH2O531z1DfegWdOw"

    base_video_url = 'https://www.youtube.com/watch?v='
    base_search_url = 'https://www.googleapis.com/youtube/v3/search?'

    first_url = base_search_url+'key={}&channelId={}&part=snippet,id&type=video&videoDuration=long&publishedAfter={}&publishedBefore={}&order=date&&maxResults=50'.format(api_key, channel_id,init_date,end_date)

    videos_metadata = []
    url = first_url
    chanel_name = None
    total_results = None
    while True:
        inp = request.urlopen(url)
        resp = json.load(inp)

        if total_results is None:
            total_results = resp['pageInfo']['totalResults']

        for i in resp['items']:
            #if i['id']['kind'] == "youtube#video":
            videos_metadata.append(i)

            if chanel_name is None:
                chanel_name = i['snippet']['channelTitle']

        print("downloaded from {} {}/{}".format(chanel_name,len(videos_metadata),total_results))
        try:
            next_page_token = resp['nextPageToken']
            url = first_url + '&pageToken={}'.format(next_page_token)
        except:
            break

    return videos_metadata


def main():
    days_to_substract = 15
    min_year = 2009
    min_month = 12
    now_local_time= datetime(day=12,month=11,year=2017)

    #local_time = now_local_time.astimezone()

    end_date = now_local_time
    videos_metadata = []
    init_date = end_date - timedelta(days_to_substract)
    while not(init_date.year<=min_year and init_date.month<=min_month):
        init_date = end_date - timedelta(days=days_to_substract)
        init_date_str = init_date.isoformat()+"Z"
        end_date_str = end_date.isoformat()+"Z"
        print("Donwloading from {} to {}".format(init_date_str,end_date_str))
        videos_metadata += get_all_video_in_channel("UChh-akEbUM8_6ghGVnJd6cQ",init_date_str,end_date_str)
        print("Downloaded from {} {}".format("UChh-akEbUM8_6ghGVnJd6cQ", len(videos_metadata)))
        end_date = init_date


    with open(DOWNLOAD_METADATA_PATH, 'wb') as f:
        pickle.dump(videos_metadata, f)

    for video in videos_metadata:
        try:
            print("https://www.youtube.com/watch?v={}".format(video["id"]["videoId"]))
        except:
            pass

    print("Finally we have download {} videos".format(len(videos_metadata)))



if __name__ == "__main__":
    main()