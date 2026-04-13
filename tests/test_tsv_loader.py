from synapsekit.loaders.tsv import TSVLoader



def test_tsv_loader():
    file_path = "test.tsv"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("name\tage\nAlice\t25\nBob\t30")

    loader = TSVLoader(file_path)
    docs = loader.load()

    assert len(docs) == 3
    assert docs[0].text == "name age"