import DocumentClassifier
import sys

if __name__ == "__main__":
    documentclassifier = DocumentClassifier.DocumentClassifier()
    print(documentclassifier.predict(sys.argv[1]))