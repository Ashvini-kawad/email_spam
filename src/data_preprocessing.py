import os
from log_file import log_function
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# Download required NLTK resources once
nltk.download('stopwords')
nltk.download('punkt_tab')
logger = log_function('data_preprocess')

# Load stopwords only once (optimization)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def sanitize_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing,
    removing stopwords and punctuation, filtering non-alphanumeric tokens,
    and applying stemming.
    """

    # Convert text to lower case
    text = text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

    # Keep only alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]

    # Apply stemming
    tokens = [ps.stem(word) for word in tokens]

    # Join back into a single string
    return " ".join(tokens)



def preprocess_data(df,text_column='text',target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(sanitize_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_data(train_data, text_column, target_column)
        test_processed_data = preprocess_data(test_data, text_column, target_column)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim_clean")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
