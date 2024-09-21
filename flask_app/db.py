import requests

url = 'https://api.dbpedia-spotlight.org/en/annotate'
text = "advancements in the field of ramjet development in the field of fluid dynamics"
params = {
    'text': text,
    'confidence': 0.5,
    'support': 20
}
headers = {
    'Accept': 'application/json'
}
response = requests.post(url, headers=headers, data=params,verify=False)
if response.status_code == 200:
    data = response.json()
    resources = data.get('Resources', [])
    impactful_keywords = []
    for resource in resources:
        surface_form = resource['@surfaceForm']
        similarity_score = float(resource['@similarityScore'])
        support = int(resource['@support'])
        
        impactful_keywords.append((surface_form, similarity_score, support))
    impactful_keywords = sorted(impactful_keywords, key=lambda x: x[1], reverse=True)
    print("Impactful Keywords:")
    for keyword, score, support in impactful_keywords:
        print(f"Keyword: {keyword}, Similarity Score: {score}, Support: {support}")
else:
    print(f"Error: {response.status_code}")
