import time
import requests
import threading
import json
import logging
import sys
from transformer import HeaderGenerator


log = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class ServerFuzzer(threading.Thread):
    """Fuzzes HTTP server by generating headers using ML engine"""
    def __init__(self, url, model, maxRequests=100):
        super().__init__()
        self.baseUrl = url
        self.model = model
        self.requestInterval = 1 # In seconds
        self.maxRequests = maxRequests
        

    def request(self, method, endpoint, headers):
        """Make HTTP request and return response"""

        payload = {'some': 'data'}
        url = self.baseUrl + endpoint

        log.info(f"Performing {method} request to: {url}")

        if method.upper() == "GET":
            return requests.get(url, headers=headers)
        elif method.upper() == "POST":
            return requests.post(url, data=json.dumps(payload), headers=headers)
        else:
            raise Exception(f"Request type '{method}' not implemented") 


    def headersFromResponse(self, resp):
        """Obtain headers from response and format them"""

        prefix = f"HTTP/1.1 {resp.status_code} {resp.reason}\n"
        headers = "\n".join(f"{k}: {v}" for k, v in resp.headers.items())
        return prefix + headers
    

    def parseModelHeaders(self, headers:list) -> dict:
        """Transform model output to dict headers"""

        parsed = {}
        headers = "".join(headers).split("\n")
        for h in headers:
            (hName, hArg) = tuple(h.split(":", 1))
            parsed[hName] = hArg
        return parsed


    def genRequestHeaders(self, headersText) -> (str, str, dict):
        """Pass response headers to ML engine and return generated request headers"""

        log.debug(f"Generating request headers using '{self.model.__name__}' model")

        predictedHeaders = self.model.translate(headersText)[1:-1] 
        log.debug(f"Predicted request headers: {predictedHeaders}")

        method = predictedHeaders[0].upper()
        endpoint = "" if predictedHeaders[1] == "/" else predictedHeaders[1]
        
        headers = self.parseModelHeaders(predictedHeaders[4:])
        return method, endpoint, headers


    def similar(self, response1:requests.Response, response2:requests.Response):
        """Checks similarity of headers to previous one"""

        if not (response1 != None and response2 != None):
            return False
        
        headers1 = self.headersFromResponse(response1)
        headers2 = self.headersFromResponse(response2)
        log.debug(f"headers1 len {len(headers1)}, headers2 len {len(headers2)}")

        # TODO: More sophisticated. Calculate some score
        return True if len(headers2) == len(headers1) else False
    

    def run(self):
        headers, prevResponse = None, None
        method, endpoint = "GET", ""

        while self.maxRequests:
            self.maxRequests -= 1

            # Initial request launches with GET method to base URL
            try:
                newResponse = self.request(method, endpoint, headers)
            except Exception as e:
                log.error(f"Cannot perform request: {e}")
                return

            log.info(f"Response. Status: {newResponse.status_code} {newResponse.reason}")

            if self.similar(newResponse, prevResponse):
                log.info("Stop fuzzing. Current response is similar to previous")
                break
            else:
                prevResponse = newResponse

            headers = self.headersFromResponse(newResponse)

            # Obtain parameters for next request
            method, endpoint, headers = self.genRequestHeaders(headers)

            log.debug(f"Waiting {self.requestInterval} seconds...")
            time.sleep(self.requestInterval)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)


    tm = HeaderGenerator(
        modelPath="./save/request_model_2551_10.pth.tar",
        srcVocPath="./save/srcVocab.pcl", 
        trgVocPath="./save/trgVocab.pcl"
    )

    serverFuzzer = ServerFuzzer(url="http://google.com", model=tm) 

    # resp = serverFuzzer.request("POST", "/", None)
    # #print("Response. Headers len:", len(resp.headers))

    # #print("Similarity:", serverFuzzer.similarity(resp, None))

    # respHeaders = serverFuzzer.headersFromResponse(resp)
    # #print("Formatted headers:", respHeaders)

    # method, endpoint, headers = serverFuzzer.genRequestHeaders(respHeaders)
    # print("Generated request method:", method)
    # print("Generated request endpoint:", endpoint)
    # print("Generated request headers:", headers)

    serverFuzzer.run()