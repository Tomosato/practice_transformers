import DocumentClassifier
import sys

if __name__ == "__main__":
    if sys.argv[1] == "j":
        documentclassifier = DocumentClassifier.DocumentClassifierForJapanese()
        print(documentclassifier.predict(sys.argv[2]))
    else:
        documentclassifier = DocumentClassifier.DocumentClassifier()
        print(documentclassifier.predict(sys.argv[2]))