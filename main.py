import sys
import logging
import argparse
from fuzzer import ServerFuzzer
from transformer import HeaderGenerator


def fixUrl(url):
    if not (url.startswith('http://') or url.startswith('https://')):
        url = 'http://' + url
    if not url.endswith('/'):
        url = url + '/'
    return url


def getUrls(path):
    with open(path, 'r') as f:
        urls = f.read()
    rawUrls = urls.split("\n")
    return [fixUrl(url) for url in rawUrls]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url')
    parser.add_argument('-f', '--file')
    parser.add_argument('-d', '--debug', nargs='?', default='default')
    args = parser.parse_args()

    # Initialize model, it will be passed to fuzzer
    tm = HeaderGenerator(
        modelPath="./save/request_model_2551_10_inf.pth.tar",
        srcVocPath="./save/srcVocab.pcl", 
        trgVocPath="./save/trgVocab.pcl"
    )

    if args.debug == "default":
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s')

    if args.url:
        serverFuzzer = ServerFuzzer(url=fixUrl(args.url), model=tm) 
        serverFuzzer.run()

    elif args.file:
        urls = getUrls(args.file)
        # fuzzers = []

        for url in urls:
            serverFuzzer = ServerFuzzer(url=url, model=tm) 
            serverFuzzer.run()
            # serverFuzzer.start()
            # fuzzers.append(serverFuzzer)
        
        # for f in fuzzers:
        #     f.join()
        
    else:
        parser.print_help()
    