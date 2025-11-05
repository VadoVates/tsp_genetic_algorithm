"""
app.py – punkt wejścia. Parsuje argumenty/konfigurację,
wybiera instancję (np. data/att48.tsp), uruchamia algorytm,
zapisuje wyniki do results/, wywołuje wizualizację.
"""

import streamlit as st
st.write("""
         ### 4.1 Wymagania

- Należy zainstalować Dockera (nie musi być to wersja Desktop) zgodnie z instrukcjami z linka: https://docs.docker.com/get-started/get-docker/
- Git
```bash
# Ubuntu/Debian
sudo apt install git
```

```bash
# Windows
https://git-scm.com/downloads/win
```
- (Opcjonalnie) `make` do wygodnego zarządzania
```bash
# Ubuntu/Debian
sudo apt install make
```
         """)