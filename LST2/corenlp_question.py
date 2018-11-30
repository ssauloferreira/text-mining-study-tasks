from stanfordcorenlp import StanfordCoreNLP
import json

nlp = StanfordCoreNLP("C:\stanford-corenlp-full-2018-02-27")
arq = open("Corpus_en_NER.txt", "r")
sentence = arq.read()

print("----- PIPELINE ------")
print('[INFO] Tokenize:', nlp.word_tokenize(sentence))

props = {'annotators': 'tokenize,ssplit', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
sentence_splitting = nlp.annotate(sentence, properties=props)
jsonToPython = json.loads(sentence_splitting)
sentences = []
for sent in jsonToPython["sentences"]:
    start_offset = sent['tokens'][0]['characterOffsetBegin']  # Begin offset of first token.
    end_offset = sent['tokens'][-1]['characterOffsetEnd']  # End offset of last token.
    sent_str = sentence[start_offset:end_offset]
    sentences.append(sent_str)
print('[INFO] Sentence Splitting:', sentences)

pos = nlp.pos_tag(sentence)
print('[INFO] Part of Speech:', pos)

props = {'annotators': 'tokenize,lemma', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
lemma = nlp.annotate(sentence, properties=props)
jsonToPython = json.loads(lemma)
lemmas = {}
for sent in jsonToPython["sentences"]:
    for token in sent['tokens']:
        word_original = token['word']
        lemma_original = token['lemma']
        if word_original not in lemmas:
            lemmas[word_original] = lemma_original
print('[INFO] Lemmatization:', lemmas)

entities = nlp.ner(sentence)
print('[INFO] Named Entities:', entities)
print('[INFO] Dependency Parsing:', nlp.dependency_parse(sentence))


print('----------------------------------------------------------------------------------------------------------------------------------------------\n'
      + '1 QUESTION')
question1 = []
for world in pos:
    if 'VB' in world[1]:
        world_aux = list(world)
        world_aux.append(lemmas[world[0]])
        question1.append(world_aux)
print("Verbos e seus lemas: ", question1)

past_form = []
for world in pos:
    if 'VBD' in world[1] or 'VBN' in world[1]:
        world_aux = list(world)
        world_aux.append(lemmas[world[0]])
        past_form.append(world_aux)
print("Verbos no passado e seus lemas: ", past_form)
print("O lemma é o verbo no presente.")


print('----------------------------------------------------------------------------------------------------------------------------------------------\n'
      + '2 QUESTION')
question2 = []
for entiti in entities:
    if 'O' not in entiti[1]:
        question2.append(entiti)
print(question2)
arq.close()


print('----------------------------------------------------------------------------------------------------------------------------------------------\n'
      + '3 QUESTION')
arq = open("NLP.txt", "r")
sentence = arq.read()
dependency = nlp.dependency_parse(sentence)
types = []
for dependences in dependency:
    if dependences[0] not in types:
        types.append(dependences[0])
print(types)
arq.close()

nlp.close()