from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

def run_baseline(train_df, test_df):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])

    # نستخدم الانحدار اللوجستي مع أوزان الفئات
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')
    clf.fit(X_train, train_df['label'])
    preds = clf.predict(X_test)

    # مقاييس التقييم الترتيبية (Ordinal Metrics)
    acc = accuracy_score(test_df['label'], preds)
    macro_f1 = f1_score(test_df['label'], preds, average='macro')
    qwk = cohen_kappa_score(test_df['label'], preds, weights='quadratic')
    mae = mean_absolute_error(test_df['label'], preds)
    
    return {'Accuracy': acc, 'Macro_F1': macro_f1, 'QWK': qwk, 'MAE': mae}
