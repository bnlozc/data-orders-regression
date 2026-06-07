#!/usr/bin/env python
# coding: utf-8

# # Siparişler - `review_score` Çok Değişkenli Regresyonu

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ⚠️ Devam etmeden önce:
# * 💾 Önceki ünitenin Siparişler çözümünü indirin
# * 👥 `order_solution.py`'nin içeriğini `olist/order.py` dosyasına kopyalayıp yapıştırın

# ⚠️ `olist` havuzunuzda `order.py` dosyasındaki kod değişikliklerini commit etmeyi unutmayın!

# 👇 Öncelikle `orders` veri setini içeri aktarın

# In[ ]:


from olist.order import Order
orders = Order().get_training_data(with_distance_seller_customer=True)


# Önceki analizimizi hatırlayalım:
# 
# Aşağıdaki 👇 korelasyon matrisi üzerinden, `review_score`'un çoğunlukla iki özellik ile ilişkili olduğunu görebiliriz: `wait_time` ve `delay_vs_expected`. Ancak, bu iki özellik de birbirleriyle yüksek oranda ilişkilidirler.
# 
# Bu alıştırmada, bir özelliğin etkisini ayırt etmek için `statsmodels` kullanacağız, **diğer özellikleri sabit tutarak**.

# In[ ]:


plt.figure(figsize = (10, 10))

sns.heatmap(
    orders.corr(numeric_only=True),
    cmap='coolwarm',
    annot=True,
    annot_kws={"size": 10}
);


# ## 1 - Tek Değişkenli Regresyon

# ❓ [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.formula.api.ols.html) ile `statsmodels.formula.api` kullanarak hızlı bir şekilde şunları oluşturun:
#  - `model1`: `review_score`'un `wait_time` üzerinde ols regresyonu
#  - `model2`: `review_score`'un `delay_vs_expected` üzerinde ols regresyonu
# 
# Her biri için `summary` tablosunu yazdırın ve sonuçları yorumlayin:
# - Bu sonuçların seaborn regplot sonuçları ile nasıl eşleştiğini anladığınızdan emin olun
# - Regresyon performans metriği `R-squared`, `individual regression coefficients`, `t-values`, `p-values` ve `95% confidence intervals`'ı okuyun

# ***Model 1***:

# In[ ]:


# YOUR CODE HERE


# ***Model 2***:

# In[ ]:


# YOUR CODE HERE


# ## 2 - Çok Değişkenli Regresyon

# ❓ Siparişe bir gün `delay_vs_expected` eklemenin `review_score` üzerindeki etkisi nedir, **`wait_time` sabit tutarak**? İki özellikten hangisi düşük `review_score` için en açıklayıcıdır?
# 
# Bu amaçla, `wait_time` ve `delay_vs_expected`'in özellikler (bağımsız değişkenler) olduğu ve `review_score`'un hedef (bağımlı değişken) olduğu bir OLS modeli `model3`'ü çalıştırın

# ***Model 3***:

# In[ ]:


# YOUR CODE HERE


# ----
# 👉 Çok değişkenli regresyon, bir özelliğin etkisini izole etmek ve diğer özelliklerin etkisini kontrol etmek için bize izin verir. Bu yeni katsayılara **`kısmi korelasyon katsayıları`** denir. 
# 
# ❓ Yukarıda *seaborn* ile hesaplanan **basit regresyon** katsayıları ile fark görebilir misiniz? 
# 
# ❓ `wait_time` ve `delay_vs_expected` için göreceli eğimler hakkında ne söyleyebilirsiniz?

# <details>
#     <summary>- 💡 Solution 💡-</summary>
# 
# - Holding `wait_time` constant, each additional day of `delay` reduces the review_score on average by 0.0205 [0.023 - 0.018] points
# - Holding `delay` constant, each additional day of `wait_time` reduces the review_score on average by 0.0383 [0.039 - 0.037] points
# 
# Contrary to what was found with the simple bivariate correlation analysis, `delay` is actually less impactful than `wait_time` in driving lower `review_score`! This interesting finding demonstrates the importance of multi-variate regression to remove the potential impact of confounding factors

# ---
# ❌ R-squared oldukça düşük: review_score varyasyonlarının en fazla %12'si `wait_time` ve `delay_vs_expected`'in birleşik varyasyonları tarafından açıklanır.
# 
# ✅ Doğrusal regresyonun açıklanabilirliğini iyileştirmek için regresyonumuza daha fazla özellik eklemeyi deneyelim.
# 
# Sonraki adımlarda `orders` veri setinden daha fazla özellik içeren yeni bir OLS `model4` oluşturacağız. Adım adım size rehberlik edeceğiz.

# 📝 <u>Not</u>: bir **`Çok Değişkenli Doğrusal Regresyon`** aynı zamanda **`Ordinary Least Squares`** yöntemi olarak da adlandırılır çünkü bu modelde **`MSE Ortalama Kare Hatalar`**'ı minimize etmek istiyoruz

# ***Model 4***:

# ❓ Hangi özellikleri göz önünde almak istersiniz?
# 
# 👉 Bu özelliklerle bir `features` DataFrame oluşturun.
# 
# - ⚠️ **veri sızıntısı** oluşturmayın: `review_score`'tan doğrudan türetilen özellikleri eklemeyin
# - ⚠️ Birbirleriyle mükemmel bir şekilde ilişkili olan iki özellik eklemeyin

# In[ ]:


# YOUR CODE HERE


# Sonra, özellikleri "standardize" edeceğiz.
# 
# **Bu ne anlama gelir?**
# 
# ⚖️  Her özellik $X_i$'yi ilgili z-score'una $Z_i = \frac{X_i - \mu_i}{\sigma_i}$ dönüştüreceğiz.
# 
# **Neden?**
# 
# Bir `çok değişkenli doğrusal regresyon`'da, her özelliğin hedef üzerindeki etkisini ölçmeye çalışıyoruz.
#     
# ⚠️ Ölçek etkileri nedeniyle, bazı özellikler yanlışlıkla diğerlerinden daha önemli olarak görülecektir
#     
# > Örneğin: "yatak odası sayısı" özelliği 1 ile 5 arasında ve "yüzey" özelliği 20 ile 200 m² arasında ise, bu iki özelliğin aralıkları oldukça farklıdır...
# 
# Bu nedenle kısmi regresyon katsayıları $\beta_i$'yi birlikte karşılaştırabilmek için standardize ederiz. Aksi takdirde, $\beta_i$ aynı boyuta sahip olmayacak, yani elmalar (örn. "gün başına review-yıldızlar") ile portakalları (örn. "BRL başına review-yıldızlar") karşılaştırmış olacaksınız!
# 
# 📅 Makine Öğrenmesi Algoritmalarını keşfettiğimizde, herhangi bir hedefi tahmin etmeye çalışmadan önce bazı ölçekleme işlemleri gerçekleştirmemiz gerekecektir.
# 
# 👉  Özelliklerinizi standardize edin:
# 
# 1. `features` DataFrame'inizden başlayın
# 1. `mean`'i çıkarın
# 1. Standart sapma `std`'ye bölün.
# 1. Sonucu `orders_standardized` değişkenine kaydedin.
# 1. Son olarak, `orders['review_score']` sütununu `orders_standardized`'e kopyalayın.
# 
# 2. ve 3. adımları basit bir matematiksel işlem olarak yazabilirsiniz, hatta DataFrames ile de. Pandas bunu anlamak için yeterince akıllıdır.
# 
# <details>
#   <summary><i>İpucu</i></summary>
# 
#   ```python
#   features - features.mean() / features.std()
#   ```
# 
# </details>

# 👉 `model4`'ü oluşturun ve eğitin.

# In[ ]:


# YOUR CODE HERE


# ---
# ❓ En önemli özellikler nelerdir? (onları iyi görselleştirmek için `.plot(kind='barh')` ile bir çubuk grafik yapın)
# - Genel regresyon performansı nasıl değişti?
# - Bu regresyon istatistiksel olarak anlamlı mıdır?

# In[ ]:


# YOUR CODE HERE


# <details>
#     <summary>- 💡 Açıklamalar 💡 -</summary>
#     
# 
# - `wait_time` en büyük açıklayıcı değişkendir
# - Tek bir sipariş için ne kadar çok `items` ve `sellers` varsa, `review_score` o kadar düşük görünür
# - Mesafe de müşteri memnuniyetinde rol oynar.
# - Özellik seçiminize bağlı olarak, `price` ve `freight_value` hakkında p-değerleri çok yüksekse hiçbir sonuca varamayabilirsiniz
#     
# - Genel olarak, bu çok değişkenli regresyon, F-istatistiği 1'den çok daha büyük olduğu için istatistiksel olarak anlamlı kalır (en az bir özelliğin çok düşük p-değeri vardır)
# 
# - R-squared çok fazla artmadı. `review_score`'un açıklanabilirliğinin çoğu orders veri seti dışında yer alır.
# 
# ⚠️ Gözlem sayısı (n) özellik sayısından (p) çok daha yüksek olduğunda düşük R-squared yaygındır. Böyle regresyonlardan, istatistiksel olarak anlamlı olması koşuluyla yine de ilgili içgörüler elde edilebilir.
# </details>
# 
# 

# ## 3 - Model Performansını Kontrol Edin

# ⚠️ Regresyon performansı sadece R-squared'i ile ölçülmemelidir!
# 
# 👀 Her zaman tahminlerin dağılımını, özellikle de kalıntıları görselleştirin.
# 
# ❓ Kalıntıları hesaplayın.
# 
# Kalıntıların ortalamasının 0'a eşit olduğunu görmelisiniz (doğrusal regresyon yapılırken bunu her zaman doğrulayın)

# In[ ]:


# YOUR CODE HERE


# 🧮 İlişkili RMSE'yi hesaplayın.

# In[ ]:


# YOUR CODE HERE


# 📊 `residuals`'ı bir histogramda çizin.

# In[ ]:


# YOUR CODE HERE


# ❓ Kalıntıların distplot'unun neden bu kadar garip bir şekle sahip olduğunu tahmin edebilir misiniz?
# 
# *İpucu:*<br/>
# 👉 Aynı grafik üzerinde hem `review_score`'un dağılımını hem de `predicted_review_score`'un dağılımını çizin.

# In[ ]:


# YOUR CODE HERE


# 📈 Aslında, önceki zorlukta `review_score`'un `delay_vs_expected`'e karşı `regresyon çizgisi`'ni zaten çizmiştik.
# 
# Bu çizimi yeniden görselleştirmek için aşağıdaki hücreyi çalıştırın:

# In[ ]:


sample = orders.sample(10000, random_state=42)
plt.figure(figsize=(13,5))
plt.suptitle('Regression of review_score, 95% confidence interval')
plt.subplot(1,2,1)
sns.regplot(x = sample.wait_time, y= sample.review_score, y_jitter=.1, ci=95)
plt.xlim(right=70)
plt.ylim(bottom=0)

plt.subplot(1,2,2)
sns.regplot(x = orders.delay_vs_expected, y= orders.review_score, y_jitter=.1, ci=95)
plt.xlim(right=70)
plt.ylim(bottom=0)


# ☝️ Bir siparişin `review_score`'unu onun `wait_time` veya `delay_vs_expected`'ine karşı regrese etmenin zor olduğunu görebilirsiniz çünkü `review_score` ayrı bir sayı olup aynı zamanda bir kategori olarak da yorumlanabilir: 1 (çok kötü), 2 (kötü), 3 (orta), 4 (iyi), 5 (mükemmel).

# 📅 Sonraki ünitede, adına rağmen bir `Classification Algorithm` olan yeni bir model keşfedeceksiniz: `Logistic Regression` 

# ☝️ Sonuç olarak, modelimiz iki nedenden dolayı o kadar harika değil:
# - İlk olarak, review_scores'un önemli bir kısmını açıklamak için yeterli özelliğimiz olmadığı için (düşük R-squared)
# - İkinci olarak, "doğrusal regresyon" işlevini ayrı bir sınıflandırma problemine uydurmaya çalıştığımız için

# 💡 Bireysel siparişler üzerinde çalışmak ve `wait_time` tabanlı ayrı `review_score`'ları açıklamaya çalışmak yerine, bir sonraki zorlukta siparişleri satıcılar tarafından toplayarak analiz edilmesini satıcı seviyesine yoğunlaştıracağız.

# 🏁 Harika iş!
# 
# 💾 İşiniz bittiğinde bu not defterini *kaydetmeyi*, *commit* ve *push* etmeyi unutmayın!
