import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Analisis & Prediksi Penjualan Mobil BMW",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Analisis & Prediksi Tingkat Penjualan Mobil BMW")
st.markdown("""
Aplikasi ini melakukan:
1. **📊 Analisis Data Historis** - Memahami pola penjualan
2. **🤖 Prediksi Klasifikasi** - High vs Low sales
3. **🎯 Rekomendasi Produksi** - Model mobil yang paling prospektif
""")

# Load data and create Engine_Size_Category column
@st.cache_data
def load_data():
    df = pd.read_excel('data\BMW-sales-data-_2010-2024_-_1_.xlsx')
    
    # Create Engine_Size_Category for all menus
    df['Engine_Size_Category'] = pd.cut(
        df['Engine_Size_L'],
        bins=[0, 2, 3, 4, 5],
        labels=['Small (<2L)', 'Medium (2-3L)', 'Large (3-4L)', 'Very Large (>4L)']
    )
    
    return df

df = load_data()

# Sidebar
st.sidebar.header("Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["📊 Overview Data", "🔍 Eksplorasi Data", "📈 Analisis Rekomendasi", 
     "🤖 Model & Prediksi", "📊 Evaluasi Model", "🎯 Rekomendasi Produksi"]
)

if menu == "📊 Overview Data":
    st.header("📊 Overview Dataset")
    
    st.subheader("Preview Data")
    st.dataframe(df.head(), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Data", f"{len(df):,}")
    with col2:
        st.metric("Jumlah Model", f"{df['Model'].nunique()}")
    with col3:
        st.metric("Range Tahun", f"{df['Year'].min()}-{df['Year'].max()}")
    
    st.subheader("Distribusi Model")
    top_models = df['Model'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_models.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('10 Model Terbanyak dalam Dataset')
    ax.set_xlabel('Model')
    ax.set_ylabel('Jumlah Data')
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif menu == "🔍 Eksplorasi Data":
    st.header("🔍 Eksplorasi Data")
    
    tab1, tab2, tab3 = st.tabs(["Distribusi", "Analisis Model", "Analisis Region"])
    
    with tab1:
        st.subheader("Distribusi Target (Sales_Classification)")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            df['Sales_Classification'].value_counts().plot(
                kind='bar', ax=ax, color=['green', 'red']
            )
            ax.set_title('Distribusi High vs Low Sales')
            ax.set_xlabel('Klasifikasi Penjualan')
            ax.set_ylabel('Jumlah')
            st.pyplot(fig)
        
        with col2:
            high_count = df['Sales_Classification'].value_counts()['High']
            low_count = df['Sales_Classification'].value_counts()['Low']
            total = len(df)
            
            st.metric("Persentase HIGH", f"{(high_count/total)*100:.1f}%")
            st.metric("Persentase LOW", f"{(low_count/total)*100:.1f}%")
    
    with tab2:
        st.subheader("Analisis Per Model")
        
        selected_model = st.selectbox(
            "Pilih Model untuk Analisis Detail:",
            df['Model'].unique()
        )
        
        model_data = df[df['Model'] == selected_model]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_sales = model_data['Sales_Volume'].mean()
            st.metric(f"Rata-rata Penjualan {selected_model}", f"{avg_sales:,.0f}")
        with col2:
            high_rate = (model_data['Sales_Classification'] == 'High').mean() * 100
            st.metric("Persentase HIGH", f"{high_rate:.1f}%")
        with col3:
            avg_price = model_data['Price_USD'].mean()
            st.metric("Harga Rata-rata", f"${avg_price:,.0f}")
        
        # Trend tahunan
        fig, ax = plt.subplots(figsize=(10, 6))
        yearly_sales = model_data.groupby('Year')['Sales_Volume'].mean()
        yearly_sales.plot(kind='line', marker='o', ax=ax)
        ax.set_title(f'Trend Penjualan {selected_model} per Tahun')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Rata-rata Volume Penjualan')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Analisis Per Region")
        
        region_data = df.groupby('Region').agg({
            'Sales_Volume': 'mean',
            'Price_USD': 'mean',
            'Sales_Classification': lambda x: (x == 'High').mean() * 100
        }).round(2)
        
        region_data.columns = ['Avg_Sales', 'Avg_Price', 'High_Rate(%)']
        
        st.dataframe(region_data.style.format({
            'Avg_Sales': '{:,.0f}',
            'Avg_Price': '${:,.0f}',
            'High_Rate(%)': '{:.1f}%'
        }), use_container_width=True)

elif menu == "📈 Analisis Rekomendasi":
    st.header("📈 Analisis untuk Rekomendasi Produksi")
    
    st.markdown("""
    ### Analisis ini membantu menentukan:
    1. **Model apa yang paling laris?**
    2. **Model apa yang konsisten HIGH sales?**
    3. **Karakteristik mobil yang disukai pasar?**
    """)
    
    # Analisis 1: Performa Model
    st.subheader("1. Performa Model Berdasarkan Volume Penjualan")
    
    # Hitung metrik per model
    model_performance = df.groupby('Model').agg({
        'Sales_Volume': ['mean', 'sum', 'count'],
        'Sales_Classification': lambda x: (x == 'High').mean() * 100,
        'Price_USD': 'mean'
    }).round(2)
    
    # Flatten column names
    model_performance.columns = [
        'Avg_Sales_Volume', 'Total_Sales', 'Count_Data',
        'High_Rate(%)', 'Avg_Price'
    ]
    
    # Sort by total sales
    model_performance = model_performance.sort_values('Total_Sales', ascending=False)
    
    st.dataframe(model_performance.head(15).style.format({
        'Avg_Sales_Volume': '{:,.0f}',
        'Total_Sales': '{:,.0f}',
        'Avg_Price': '${:,.0f}',
        'High_Rate(%)': '{:.1f}%'
    }), use_container_width=True)
    
    # Visualisasi
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_10_models = model_performance.head(10)
        ax.barh(top_10_models.index, top_10_models['Total_Sales'], color='skyblue')
        ax.set_xlabel('Total Penjualan')
        ax.set_title('Top 10 Model berdasarkan Total Penjualan')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        high_rate_sorted = model_performance.sort_values('High_Rate(%)', ascending=False).head(10)
        ax.barh(high_rate_sorted.index, high_rate_sorted['High_Rate(%)'], color='lightgreen')
        ax.set_xlabel('Persentase HIGH Sales (%)')
        ax.set_title('Top 10 Model berdasarkan Persentase HIGH')
        st.pyplot(fig)
    
    # Analisis 2: Karakteristik Mobil yang Sukses
    st.subheader("2. Analisis Karakteristik Mobil dengan HIGH Sales")
    
    high_sales = df[df['Sales_Classification'] == 'High']
    low_sales = df[df['Sales_Classification'] == 'Low']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Fuel Type yang paling sukses:**")
        fuel_success = high_sales['Fuel_Type'].value_counts(normalize=True) * 100
        st.dataframe(fuel_success.round(1))
    
    with col2:
        st.write("**Transmission yang paling sukses:**")
        trans_success = high_sales['Transmission'].value_counts(normalize=True) * 100
        st.dataframe(trans_success.round(1))
    
    with col3:
        st.write("**Warna yang paling populer:**")
        color_success = high_sales['Color'].value_counts(normalize=True).head(5) * 100
        st.dataframe(color_success.round(1))
    
    # Analisis 3: Engine Size vs Success Rate
    st.subheader("3. Analisis Engine Size vs Tingkat Kesuksesan")
    
    engine_success = df.groupby('Engine_Size_Category')['Sales_Classification'].apply(
        lambda x: (x == 'High').mean() * 100
    ).round(1)
    
    st.dataframe(engine_success)

elif menu == "🎯 Rekomendasi Produksi":
    st.header("🎯 Rekomendasi Model untuk Diproduksi")
    st.markdown("### Berdasarkan Analisis Data Historis 2010-2024")
    
    # Pastikan Engine_Size_Category ada di DataFrame
    if 'Engine_Size_Category' not in df.columns:
        df['Engine_Size_Category'] = pd.cut(
            df['Engine_Size_L'],
            bins=[0, 2, 3, 4, 5],
            labels=['Small (<2L)', 'Medium (2-3L)', 'Large (3-4L)', 'Very Large (>4L)']
        )
    
    # Metrik untuk scoring
    st.subheader("Metrik Penilaian Model")
    st.info("""
    **Skoring berdasarkan 4 faktor:**
    1. **Volume Penjualan Total** (Bobot: 40%)
    2. **Persentase HIGH Sales** (Bobot: 30%)
    3. **Konsistensi Trend** (Bobot: 20%)
    4. **Harga Rata-rata** (Bobot: 10%) - Lebih tinggi lebih baik
    """)
    
    # Hitung skor untuk setiap model
    model_stats = df.groupby('Model').agg({
        'Sales_Volume': ['sum', 'mean', 'std'],
        'Sales_Classification': lambda x: (x == 'High').mean() * 100,
        'Price_USD': 'mean',
        'Year': 'count'
    })
    
    # Flatten columns
    model_stats.columns = [
        'Total_Sales', 'Avg_Sales', 'Std_Sales',
        'High_Rate', 'Avg_Price', 'Data_Count'
    ]
    
    # Normalisasi setiap metrik (0-100)
    def normalize(series):
        if series.max() == series.min():
            return pd.Series([50] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min()) * 100
    
    model_stats['Score_Volume'] = normalize(model_stats['Total_Sales']) * 0.4
    model_stats['Score_HighRate'] = normalize(model_stats['High_Rate']) * 0.3
    model_stats['Score_Consistency'] = (100 - normalize(model_stats['Std_Sales'])) * 0.2  # Std rendah = konsisten
    model_stats['Score_Price'] = normalize(model_stats['Avg_Price']) * 0.1
    
    # Total skor
    model_stats['Total_Score'] = (
        model_stats['Score_Volume'] +
        model_stats['Score_HighRate'] +
        model_stats['Score_Consistency'] +
        model_stats['Score_Price']
    )
    
    # Urutkan berdasarkan skor
    model_stats = model_stats.sort_values('Total_Score', ascending=False)
    
    # Tampilkan top 10 rekomendasi
    st.subheader("🏆 Top 10 Rekomendasi Model untuk Diproduksi")
    
    top_10 = model_stats.head(10).copy()
    top_10['Rank'] = range(1, 11)
    
    # Format untuk display
    display_df = top_10[['Total_Score', 'Total_Sales', 'High_Rate', 'Avg_Price']].round(2)
    display_df.columns = ['Total Skor', 'Total Penjualan', '% HIGH Sales', 'Harga Rata-rata']
    
    # Reset index untuk tampilkan Model
    display_df = display_df.reset_index()
    display_df = display_df[['Model', 'Total Skor', 'Total Penjualan', '% HIGH Sales', 'Harga Rata-rata']]
    
    # Tabel dengan warna
    st.dataframe(
        display_df.style.format({
            'Total Skor': '{:.1f}',
            'Total Penjualan': '{:,.0f}',
            '% HIGH Sales': '{:.1f}%',
            'Harga Rata-rata': '${:,.0f}'
        }).apply(
            lambda x: ['background-color: gold' if x.name == 0 
                      else 'background-color: silver' if x.name == 1
                      else 'background-color: #cd7f32' if x.name == 2
                      else '' for i in x],
            axis=1
        ),
        use_container_width=True
    )
    
    # Visualisasi
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(display_df['Model'][:5], display_df['Total Skor'][:5], color=['gold', 'silver', '#cd7f32', 'lightblue', 'lightblue'])
        ax.set_xlabel('Total Skor')
        ax.set_title('Top 5 Model Terbaik')
        st.pyplot(fig)
    
    with col2:
        # Analisis tahun terakhir (2024) untuk melihat trend terbaru
        latest_data = df[df['Year'] >= 2020]  # Data 4 tahun terakhir
        if not latest_data.empty:
            recent_performance = latest_data.groupby('Model')['Sales_Classification'].apply(
                lambda x: (x == 'High').mean() * 100
            ).sort_values(ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            recent_performance.plot(kind='barh', ax=ax, color='green')
            ax.set_xlabel('Persentase HIGH Sales (2020-2024)')
            ax.set_title('Perform Terbaru (4 Tahun Terakhir)')
            st.pyplot(fig)
    
    # Rekomendasi berdasarkan segmentasi
    st.subheader("📋 Rekomendasi Berdasarkan Segmentasi")
    
    tab1, tab2, tab3 = st.tabs(["Berdasarkan Region", "Berdasarkan Fuel Type", "Berdasarkan Engine Size"])
    
    with tab1:
        st.write("**Top Model per Region (berdasarkan % HIGH Sales):**")
        for region in df['Region'].unique():
            region_data = df[df['Region'] == region]
            if not region_data.empty:
                top_model = region_data.groupby('Model')['Sales_Classification'].apply(
                    lambda x: (x == 'High').mean() * 100
                ).idxmax()
                top_rate = region_data.groupby('Model')['Sales_Classification'].apply(
                    lambda x: (x == 'High').mean() * 100
                ).max()
                st.write(f"**{region}:** {top_model} ({top_rate:.1f}% HIGH rate)")
    
    with tab2:
        st.write("**Top Model per Fuel Type (berdasarkan rata-rata penjualan):**")
        for fuel in df['Fuel_Type'].unique():
            fuel_data = df[df['Fuel_Type'] == fuel]
            if len(fuel_data) > 10:  # Hanya jika ada cukup data
                best_model = fuel_data.groupby('Model')['Sales_Volume'].mean().idxmax()
                avg_sales = fuel_data.groupby('Model')['Sales_Volume'].mean().max()
                st.write(f"**{fuel}:** {best_model} (Avg Sales: {avg_sales:,.0f})")
    
    with tab3:
        st.write("**Rekomendasi berdasarkan Ukuran Mesin:**")
        for engine_cat in ['Small (<2L)', 'Medium (2-3L)', 'Large (3-4L)', 'Very Large (>4L)']:
            if engine_cat in df['Engine_Size_Category'].unique():
                engine_data = df[df['Engine_Size_Category'] == engine_cat]
                if not engine_data.empty:
                    best_model = engine_data.groupby('Model')['Sales_Classification'].apply(
                        lambda x: (x == 'High').mean() * 100
                    ).idxmax()
                    best_rate = engine_data.groupby('Model')['Sales_Classification'].apply(
                        lambda x: (x == 'High').mean() * 100
                    ).max()
                    st.write(f"**{engine_cat}:** {best_model} ({best_rate:.1f}% HIGH rate)")
    
    # Kesimpulan
    st.subheader("🎯 Kesimpulan Rekomendasi")
    
    if not display_df.empty:
        top_model = display_df.iloc[0]['Model']
        top_score = display_df.iloc[0]['Total Skor']
        
        st.success(f"""
        **REKOMENDASI UTAMA: FOKUS PADA MODEL {top_model.upper()}**
        
        **Alasan:**
        1. **Skor Terbaik:** {top_score:.1f} (dari skala 100)
        2. **Total Penjualan:** {display_df.iloc[0]['Total Penjualan']:,.0f} unit
        3. **Tingkat Kesuksesan:** {display_df.iloc[0]['% HIGH Sales']} HIGH sales
        4. **Harga Kompetitif:** ${display_df.iloc[0]['Harga Rata-rata']:,.0f}
        
        **Saran Produksi:**
        - Pertahankan produksi **{top_model}**
        - Tingkatkan kapasitas produksi untuk **{display_df.iloc[1]['Model']}** (peringkat 2)
        - Evaluasi model dengan skor terendah untuk kemungkinan penghentian
        """)
    else:
        st.warning("Tidak ada data untuk menampilkan rekomendasi.")

elif menu == "🤖 Model & Prediksi":
    st.header("🤖 Model Machine Learning & Prediksi")
    
    st.info("""
    **Fitur ini memprediksi apakah KONFIGURASI SPESIFIK suatu mobil akan memiliki penjualan HIGH atau LOW.**
    
    Gunakan untuk:
    1. Evaluasi desain baru
    2. Prediksi sebelum launch
    3. Optimasi spesifikasi
    """)
    
    # Preprocessing
    st.subheader("1. Preprocessing Data")
    
    # Encoding categorical variables
    df_encoded = df.copy()
    categorical_cols = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    
    # Encode target
    le_target = LabelEncoder()
    df_encoded['Sales_Classification'] = le_target.fit_transform(df_encoded['Sales_Classification'])
    
    # Prepare features and target
    X = df_encoded.drop(['Sales_Classification', 'Sales_Volume', 'Engine_Size_Category'], axis=1)
    y = df_encoded['Sales_Classification']
    
    st.success("✅ Preprocessing selesai! Data kategorikal telah diencode.")
    
    # Split data
    st.subheader("2. Split Data Training & Testing")
    test_size = st.slider("Persentase Data Testing (%)", 10, 40, 20) / 100
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    st.write(f"**Data Training:** {len(X_train):,} sampel ({100*(1-test_size):.0f}%)")
    st.write(f"**Data Testing:** {len(X_test):,} sampel ({test_size*100:.0f}%)")
    
    # Train model
    st.subheader("3. Training Model Random Forest")
    
    n_estimators = st.slider("Jumlah Trees", 10, 200, 100)
    max_depth = st.slider("Kedalaman Maksimum", 2, 20, 10)
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.session_state['model'] = model
            st.session_state['accuracy'] = accuracy
            st.session_state['y_pred'] = y_pred
            st.session_state['y_test'] = y_test
            st.session_state['encoders'] = encoders
            st.session_state['le_target'] = le_target
            st.session_state['X_columns'] = X.columns.tolist()
            st.session_state['categorical_cols'] = categorical_cols
        
        st.success(f"✅ Model trained successfully!")
        st.metric("Akurasi Model", f"{accuracy:.2%}")
    
    # Prediction section
    st.subheader("4. Prediksi Data Baru")
    
    if 'model' in st.session_state:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_input = st.selectbox("Model", df['Model'].unique())
            year_input = st.number_input("Tahun", 2010, 2024, 2020)
            region_input = st.selectbox("Region", df['Region'].unique())
            color_input = st.selectbox("Warna", df['Color'].unique())
        
        with col2:
            fuel_input = st.selectbox("Tipe Bahan Bakar", df['Fuel_Type'].unique())
            transmission_input = st.selectbox("Transmisi", df['Transmission'].unique())
            engine_input = st.number_input("Engine Size (L)", 1.0, 5.0, 2.0, 0.1)
            mileage_input = st.number_input("Mileage (KM)", 0, 300000, 50000, 1000)
        
        with col3:
            price_input = st.number_input("Harga (USD)", 10000, 200000, 50000, 1000)
            sales_volume_input = st.number_input("Volume Penjualan", 0, 20000, 1000, 100)
        
        if st.button("🔮 Prediksi", type="primary"):
            # Prepare input data
            input_data = pd.DataFrame({
                'Model': [model_input],
                'Year': [year_input],
                'Region': [region_input],
                'Color': [color_input],
                'Fuel_Type': [fuel_input],
                'Transmission': [transmission_input],
                'Engine_Size_L': [engine_input],
                'Mileage_KM': [mileage_input],
                'Price_USD': [price_input],
                'Sales_Volume': [sales_volume_input]
            })
            
            # Encode categorical features
            for col in st.session_state['categorical_cols']:
                if col in input_data.columns:
                    try:
                        input_data[col] = st.session_state['encoders'][col].transform([input_data[col].iloc[0]])[0]
                    except ValueError:
                        # Handle unseen labels by using most common class
                        input_data[col] = 0
            
            # Reorder columns to match training data
            input_data = input_data[st.session_state['X_columns']]
            
            # Make prediction
            prediction = st.session_state['model'].predict(input_data)[0]
            prediction_proba = st.session_state['model'].predict_proba(input_data)[0]
            
            # Decode prediction
            predicted_class = st.session_state['le_target'].inverse_transform([prediction])[0]
            
            # Display results
            st.success(f"**Prediksi:** {predicted_class}")
            
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.metric("Probabilitas HIGH", f"{prediction_proba[1]:.2%}")
            with col_pred2:
                st.metric("Probabilitas LOW", f"{prediction_proba[0]:.2%}")
            
            # Feature importance
            st.subheader("📊 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': st.session_state['X_columns'],
                'Importance': st.session_state['model'].feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['Feature'], feature_importance['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importance')
            st.pyplot(fig)

elif menu == "📊 Evaluasi Model":
    st.header("📊 Evaluasi Model")
    
    if 'model' in st.session_state:
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(
            st.session_state['y_test'], 
            st.session_state['y_pred'],
            target_names=['Low', 'High'],
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Low', 'High'], 
                    yticklabels=['Low', 'High'],
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{st.session_state['accuracy']:.2%}")
        with col2:
            precision = report['High']['precision']
            st.metric("Precision (High)", f"{precision:.2%}")
        with col3:
            recall = report['High']['recall']
            st.metric("Recall (High)", f"{recall:.2%}")
    
    else:
        st.warning("⚠️ Silakan train model terlebih dahulu di menu 'Model & Prediksi'")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Fitur Utama:**")
st.sidebar.markdown("✅ Prediksi Klasifikasi Penjualan")
st.sidebar.markdown("✅ Analisis Data Historis")
st.sidebar.markdown("✅ Rekomendasi Produksi")
st.sidebar.markdown("✅ Evaluasi Model Machine Learning")