import ner
tagger = ner.SocketNER(host='localhost', port=8080)
tagger.get_entities("University of California is located in California, United States")