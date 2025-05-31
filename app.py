import streamlit as st
import numpy as np
import pandas as pd
from difflib import get_close_matches
import joblib
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from catboost import CatBoostClassifier

torch.classes.__path__ = []

st.set_page_config(page_title="Подбор вакансий и оценка резюме", layout="wide")


# === Стилизация ===
st.markdown("""
    <style>
    .stTextInput, .stSelectbox, .stNumberInput, .stRadio, .stTextArea {
        padding: 5px !important;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        font-size: 1em;
    }
    </style>
""", unsafe_allow_html=True)
st.title("📄 Оценка резюме и подбор вакансий")

st.markdown("""
#### Этот инструмент использует машинное обучение для анализа вашего резюме и подбора наиболее подходящих вакансий из базы данных.

-  Получите подборку релевантных вакансий  
---
""")
col1, col2 = st.columns([1, 2])

with col1:
        st.subheader("🔧 Ввод данных")
        birthday = st.number_input("Год рождения", min_value=1950, max_value=2024, value=1995)
        experience = st.number_input("Опыт работы (лет)", min_value=0, value=2)
        salary = st.number_input("Желаемая зарплата (₽)", value=60000)
        region_text = st.text_input("Регион проживания")
        profession_text = st.text_input("Желаемая профессия", value='Не указано')
        relocation = st.selectbox("Готовность к переезду", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет")
        retraining = st.selectbox("Готовность к переобучению", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет")
        business_trips = st.selectbox("Командировки", [1, 0], format_func=lambda x: "Да" if x == 1 else "Нет")
        busy_type = st.multiselect("Тип занятости", options=['Полная занятость','Сезонная','Стажировка','Удаленная','Частичная занятость'])
        schedule_type = st.multiselect("Желаемый график работы", options=['Вахтовый метод','Гибкий график','Ненормированный рабочий график','Неполный рабочий день','Полный рабочий день','Сменный график'])
        skills = st.text_area("Навыки (в свободной форме)", value='Не указано')

        submitted = st.button("Проанализировать")

with col2:
        if submitted:
            st.subheader("Результаты анализа")

            df = pd.DataFrame(data = {'birthday':[birthday],
                               'business_trips':[business_trips],
                               'experience':[experience],
                               'relocation':[relocation],
                               'retraining_capability_x':[retraining],
                               'salary':[salary],
                               'schedule_type_1':[int('Вахтовый метод' in schedule_type)],
                               'schedule_type_2':[int('Гибкий график' in schedule_type)],
                               'schedule_type_3':[int('Ненормированный рабочий график' in schedule_type)],
                               'schedule_type_4':[int('Неполный рабочий день' in schedule_type)],
                               'schedule_type_5':[int('Полный рабочий день' in schedule_type)],
                               'schedule_type_6':[int('Сменный график' in schedule_type)],
                               'busy_type_Полная занятость':['Полная занятость' in busy_type],
                               'busy_type_Сезонная':['Сезонная' in busy_type],
                               'busy_type_Стажировка':['Стажировка' in busy_type],
                               'busy_type_Удаленная':['Удаленная' in busy_type],
                               'busy_type_Частичная занятость':['Частичная занятость' in busy_type],
                               'position_name':profession_text,
                               'skills':skills
                               })
            model_name = "DeepPavlov/rubert-base-cased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # === 2️⃣ Определяем DataLoader для обработки в батчах ===
            class TextDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length=512):
                    self.texts = [str(text)[:max_length] for text in texts]  # Обрезаем до 512 символов
                    self.tokenizer = tokenizer
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    return self.texts[idx]
            # === 3️⃣ Признаки для векторизации ===
            text_features = [
                'position_name', 
                'skills'
            ]
            # === 4️⃣ Векторизация в батчах ===
            batch_size = 32
            for feature in text_features: 
                dataset = TextDataset(df[feature].fillna("Не указано").tolist(), tokenizer)
                dataloader = DataLoader(dataset, batch_size=batch_size)
                embeddings = []
                model.eval()
                with torch.no_grad():
                    for batch in tqdm(dataloader, total=len(dataloader)):
                        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                        outputs = model(**inputs)
                        embeddings.extend(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
                bert_df = pd.DataFrame(embeddings, columns=[f"{feature}_emb_{i}" for i in range(768)])
                df = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)
            scaler = joblib.load('scaler.pkl')
            numerical_features = ['birthday', 'experience', 'salary']
            df[numerical_features] = scaler.transform(df[numerical_features])
            pca = joblib.load('pca_model.pkl')
            embedding_columns = [col for col in df.columns if 'skills_emb' in col or 'position_name_emb' in col]
            X1 = df[embedding_columns]
            X1_reduced = pca.transform(X1)
            X1_reduced_df = pd.DataFrame(X1_reduced, columns=[f'resume_emb_{i}' for i in range(300)])
            df = pd.concat([df.drop(columns=embedding_columns), X1_reduced_df], axis=1)
            df = df.drop(columns=['skills','position_name'])
            model = CatBoostClassifier()
            model.load_model("resume_quality_model.cbm")
            resume_cols = [col for col in df.columns if col.startswith('resume_emb_')]
            resume_additional = [
                'business_trips', 'experience',
                'relocation', 'retraining_capability_x', 'salary'
            ] + [col for col in df.columns if 'schedule_type' in col or 'busy_type' in col]
            resume_features = resume_cols + resume_additional
            df1 = list(df[resume_features].values)
            X_new = np.array(df1).reshape(1, -1)
            prob_success = model.predict_proba(X_new)[0, 1]
            df.to_csv('test.csv', sep=';', encoding='cp1251')
            st.success(f"Оценка успешности резюме: {round(prob_success*100,1)}%")
            st.markdown("### Рекомендуемые вакансии:")
            # === Загрузка данных ===
            df1 = pd.read_csv('for_site.csv')
            df1 = df1.drop_duplicates(subset='id_vacancy')
            # Повторим ту же логику выделения признаков
            resume_cols = [col for col in df1.columns if col.startswith('resume_emb_')]
            vacancy_cols = [col for col in df1.columns if col.startswith('vacancy_emb_')]

            resume_additional = ['birthday', 'business_trips', 'experience', 'relocation', 'retraining_capability_x', 'salary'] + \
                                [col for col in df1.columns if 'schedule_type' in col or 'busy_type' in col]

            vacancy_additional = ['accommodation_capability', 'base_salary_max', 'base_salary_min', 'disabled', 'dms',
                                'experience_requirements', 'large_families', 'single_parent',
                                'vouchers_health_institutions', 'work_places', 'workers_with_disabled_children'] + \
                                [col for col in df1.columns if 'education_type' in col or 'employment_type' in col or
                                'inner_info' in col or 'source_' in col or 'education_requirements' in col]

            resume_features = resume_cols + resume_additional
            vacancy_features = vacancy_cols + vacancy_additional

            X_resume = df[resume_features].astype(np.float32).values
            X_vacancy = df1[vacancy_features].astype(np.float32).values

            # === Определение архитектуры ===
            class ResumeVacancyModel(torch.nn.Module):
                def __init__(self, resume_dim, vacancy_dim):
                    super().__init__()
                    self.fc = torch.nn.Sequential(
                        torch.nn.Linear(resume_dim + vacancy_dim, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, 128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 1),
                        torch.nn.Sigmoid()
                    )

                def forward(self, resume_input, vacancy_input):
                    x = torch.cat((resume_input, vacancy_input), dim=1)
                    return self.fc(x).squeeze()

            # === Загрузка модели ===
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ResumeVacancyModel(X_resume.shape[1], X_vacancy.shape[1])
            model.load_state_dict(torch.load("resume_vacancy_model2.pth", map_location=device))
            model.to(device)
            model.eval()

            # === Рекомендации ===
            def recommend_vacancies(resume_vector, vacancy_matrix, model, top_n=10):
                resume_batch = np.repeat(resume_vector[np.newaxis, :], vacancy_matrix.shape[0], axis=0)
                with torch.no_grad():
                    inputs_r = torch.tensor(resume_batch, dtype=torch.float32).to(device)
                    inputs_v = torch.tensor(vacancy_matrix, dtype=torch.float32).to(device)
                    scores = model(inputs_r, inputs_v).cpu().numpy()
                top_indices = np.argsort(scores)[::-1][:top_n]
                return top_indices, scores[top_indices]

            # === Пример использования ===
            resume_example = X_resume[0]  # или замени на своё резюме
            top_idxs, top_scores = recommend_vacancies(resume_example, X_vacancy, model)

            # Показать результат
            recommended_vacancies = df1.iloc[top_idxs]
            vac = recommended_vacancies[['id_vacancy']]
            df2 = pd.read_csv('for_rec.csv')
            df2 = df2.drop_duplicates(subset='id_vacancy')
            res = df2[df2.id_vacancy.isin(vac.id_vacancy)][['title','base_salary_min','base_salary_max','job_benefits_other_benefits','job_location_address','requirements_qualifications','responsibilities','additional_info']]
            for index, row in res.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid #444; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: #1e1e1e;">
                        <h4 style="color: #4CAF50;">{row['title']}</h4>
                        <p><strong>Адрес:</strong> {row['job_location_address']}</p>
                        <p><strong>Зарплата:</strong> {row['base_salary_min']} – {row['base_salary_max']} ₽</p>
                        <p><strong>Требуемая квалификация:</strong> {row['requirements_qualifications']}</p>
                        <p><strong>Должностные обязанности:</strong> {row['responsibilities']}</p>
                        <p><strong>Дополнительная информация:</strong> {row['additional_info']}</p>
                        <p><strong>Бонусы/льготы:</strong> {row['job_benefits_other_benefits']}</p>
                    </div>
                    """, unsafe_allow_html=True)


            # for i, (title, salary_min, salary_max) in enumerate([
            #     ("Учитель английского языка", 50000, 70000),
            #     ("Методист в лицей", 60000, 85000),
            #     ("Преподаватель информатики", 55000, 75000)
            # ], 1):
            #     link = f"/vacancy?id={i}"
            #     st.markdown(f"{i}. [{title} — {salary_min}–{salary_max}₽](#{link})")

