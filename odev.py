import torch
import torch.nn as nn
import torch.optim as optim

# Rastgelelik tohumu (sonuçlar tutarlı çıksın diye)
torch.manual_seed(42)

# Sentetik Veri Üretimi
X = torch.randn(1000, 20)
y = (
    torch.matmul(X[:, :5], torch.tensor([1.5, -2.0, 0.5, 1.0, -1.0]))
    + 0.1 * torch.randn(1000)
)
y = y.unsqueeze(1)

criterion = nn.MSELoss()


# ---TEMEL MODEL (Base Network) ---
class BaseNetwork(nn.Module):

  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(20, 16)
    self.relu = nn.ReLU()
    self.output = nn.Linear(16, 1)

  def forward(self, x):
    return self.output(self.relu(self.layer1(x)))


# Taban modeli eğitiyoruz
base_model = BaseNetwork()
optimizer_base = optim.Adam(base_model.parameters(), lr=0.01)
for epoch in range(100):
  optimizer_base.zero_grad()
  loss = criterion(base_model(X), y)
  loss.backward()
  optimizer_base.step()


# Genişletilmiş Statik ve Sıralı Model İçin Yapı
class ExpandedNetwork(nn.Module):

  def __init__(self, pretrained_base):
    super().__init__()
    self.layer1 = pretrained_base.layer1
    self.relu1 = pretrained_base.relu
    self.new_layer = nn.Linear(16, 16)
    self.relu2 = nn.ReLU()
    self.output = pretrained_base.output

  def forward(self, x):
    out = self.relu1(self.layer1(x))
    out = self.relu2(self.new_layer(out))
    return self.output(out)


# Paralel + Füzyon Modeli
class ParallelFusionNetwork(nn.Module):

  def __init__(self, pretrained_base):
    super().__init__()
    # Kol 1 (Taban Akışı)
    self.layer1 = pretrained_base.layer1
    self.relu1 = pretrained_base.relu
    self.base_output = pretrained_base.output

    # Kol 2 (Paralel Yeni Katman Akışı)
    self.new_layer = nn.Linear(16, 16)
    self.relu2 = nn.ReLU()
    self.new_output = nn.Linear(16, 1)

  def forward(self, x):
    # Kol A Çıktısı
    out1 = self.relu1(self.layer1(x))
    res1 = self.base_output(out1)

    # Kol B Çıktısı (Paralel Kol)
    out2 = self.relu1(self.layer1(x))
    out2 = self.relu2(self.new_layer(out2))
    res2 = self.new_output(out2)

    # ---FUSION ---
    final_output = res1 + res2
    return final_output


# --- YÖNTEMLERİN ÇALIŞTIRILMASI VE LOSS HESAPLARI ---

# YÖNTEM 1: Statik Optimizasyon
model_static = ExpandedNetwork(base_model)
optimizer_static = optim.Adam(model_static.parameters(), lr=0.01)
for epoch in range(150):
  optimizer_static.zero_grad()
  loss = criterion(model_static(X), y)
  loss.backward()
  optimizer_static.step()
static_final_loss = criterion(model_static(X), y).item()


# YÖNTEM 2: Sıralı 2 Aşamalı Eğitim (Sequential Adapted)
model_adapted = ExpandedNetwork(base_model)
for param in model_adapted.layer1.parameters():
  param.requires_grad = False
optimizer_phase1 = optim.Adam(
    list(model_adapted.new_layer.parameters())
    + list(model_adapted.output.parameters()),
    lr=0.01,
)
for epoch in range(50):
  optimizer_phase1.zero_grad()
  loss = criterion(model_adapted(X), y)
  loss.backward()
  optimizer_phase1.step()

for param in model_adapted.layer1.parameters():
  param.requires_grad = True
optimizer_phase2 = optim.Adam([
    {"params": model_adapted.layer1.parameters(), "lr": 0.001},
    {"params": model_adapted.new_layer.parameters(), "lr": 0.005},
    {"params": model_adapted.output.parameters(), "lr": 0.005},
])
for epoch in range(100):
  optimizer_phase2.zero_grad()
  loss = criterion(model_adapted(X), y)
  loss.backward()
  optimizer_phase2.step()
adapted_final_loss = criterion(model_adapted(X), y).item()


# YÖNTEM 3: Paralel + Füzyon Optimizasyonu
model_parallel = ParallelFusionNetwork(base_model)
optimizer_parallel = optim.Adam(model_parallel.parameters(), lr=0.01)
for epoch in range(150):
  optimizer_parallel.zero_grad()
  loss = criterion(model_parallel(X), y)
  loss.backward()
  optimizer_parallel.step()
parallel_final_loss = criterion(model_parallel(X), y).item()


# --- SONUÇLARIN KARŞILAŞTIRILMASI (Rapor İçin) ---
print("\n" + "=" * 40)
print("  ÖDEV SONUÇLARI KARŞILAŞTIRMASI")
print("=" * 40)
print(f"1. Static Optimization Final Loss:      {static_final_loss:.4f}")
print(f"2. Sequential Adapted Final Loss:   {adapted_final_loss:.4f}")
print(f"3. Parallel + Fusion Final Loss:    {parallel_final_loss:.4f}")
print("=" * 40)