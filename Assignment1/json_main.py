a  = "Assignment1\sample_test.json"
import task1
# print(a)
import  json
with open(a) as f:
    data = json.load(f)
    # print(data)
    # print(type(data))
a = task1.WordPieceTokenizer(1000
, 'Assignment1\corpus.txt',r'Assignment1\vocabulary_5.txt')
a.read_karo_corpus()
# exit(0)
a.preprocess_data()
a.construct_vocabulary()

a.json_formatter(data)