from characteristics_extractor import CharactericsExtractor

# text_piped; not lemmatized
extractor = CharactericsExtractor()
full_texts = False
if full_texts:
    with open("descriptions_1.txt") as f:
        texts = f.readlines()
        texts = [t.strip().split() for t in texts]
else:
    with open("avito_test_set") as f:
        texts = f.readlines()
        texts = [[t.strip()] for t in texts]
car_characteristics = [[]] * len(texts)


# with open("descriptions_2.txt") as f:
#     texts_normed = f.readlines()
# texts_normed = [t.strip().split() for t in texts_normed]

for t_i, t in enumerate(texts):
    print("\t", t_i, end="\r")
    output = extractor.parse_text(t, piped=full_texts)
    car_characteristics[t_i] = output
    # print(output)
