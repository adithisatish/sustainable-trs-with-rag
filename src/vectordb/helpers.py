from src import * 

def create_chunks(city, country, text):
    for i, line in enumerate(text):
        if line[0] == "\n":
            del text[i]

    index = 0 
    chunks = []
    pattern = re.compile("==")
    ignore = re.compile("===")
    section = 'Introduction'
    for i, line in enumerate(text):
        if pattern.search(line) and not ignore.search(line):
            chunk = ''.join(text[index:i])
            chunks.append({
                'city': city,
                'country': country,
                'section': section,
                'text': chunk, 
                # 'vector': f'city: {city}, country: {country}, section: {section}, text: {chunk}' 
            })
            index=i+1
            section = re.sub(pattern, '', line).strip()

    df = pd.DataFrame(chunks)
    return df

def read_docs():
    df = pd.DataFrame()
    cities = pd.read_csv("../city_abstracts_embeddings.csv")
    for file_name in os.listdir(wikivoyage_docs_dir):
        city = file_name.split(".")[0]
        country = cities[cities['city'] == city]['country'].item()
        with open(wikivoyage_docs_dir + file_name) as file: 
            text = file.readlines()
            chunk_df = create_chunks(city, country, text)
            df = pd.concat([df,chunk_df])

    return df

def read_listings():
    df = pd.read_csv(wikivoyage_listings_dir + "wikivoyage-listings-cleaned.csv")
    return df

def preprocess_df(df):
    section_counts = df['section'].value_counts()
    sections_to_keep = section_counts[section_counts > 150].index
    filtered_df = df[df['section'].isin(sections_to_keep)]

    return filtered_df

def compute_wv_docs_embeddings(df):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vector_dimension = model.get_sentence_embedding_dimension()
    
    print("Computing embeddings")
    embeddings = []
    for i, row in df.iterrows():
        emb = model.encode(row['combined'], show_progress_bar=True).tolist()
        embeddings.append(emb)

    print("Finished computing embeddings for wikivoyage documents.")
    df['vector'] = embeddings
    # df.to_csv(wv_embeddings + "wikivoyage-listings-embeddings.csv")
    # print("Finished saving file.")
    return df

def embed_query(query):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # vector_dimension = model.get_sentence_embedding_dimension()   
    embedding = model.encode(query).tolist()
    return embedding