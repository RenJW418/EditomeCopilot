import requests
import json

def test_epmc():
    # Jinek et al. 2012
    doi = "10.1126/science.1225829" 
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": f'DOI:"{doi}"',
        "format": "json",
        "resultType": "core"
    }
    
    resp = requests.get(url, params=params)
    data = resp.json()
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    test_epmc()
