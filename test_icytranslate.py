# *-* coding: utf-8 *-*
import json
import base64
import argparse
import sys
import requests

def translate(sent):
    payload = {'sent':sent}
    r = requests.get("http://localhost:8842/translate/singlesentence", data=payload)
    for i in json.loads(r.text)['translates']:
        print(i)
        
def translate_multi(segment):
    payload = {'segment':segment}
    r = requests.get("http://localhost:8842/translate/segment", data=payload)
    for k,v in json.loads(r.text).iteritems():
        print(v)
        #print ("key:{},value:{}".format(k,v))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sent',type=str,help='sentence to be translated')
    args = parser.parse_args(sys.argv[1:])
    translate_multi(args.sent)