import os
import logging
import time
from azure.monitor.opentelemetry import configure_azure_monitor
from dotenv import load_dotenv

# 1. Chargement des variables
load_dotenv()

# Récupération de la clé
CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

print("-" * 50)
print("TEST DE MONITORING AZURE")
print("-" * 50)

if not CONNECTION_STRING:
    print("❌ ERREUR: La variable d'environnement n'est pas trouvée/vide.")
    exit()

print(f"🔑 Clé trouvée (début) : {CONNECTION_STRING[:30]}...")

try:
    # 2. Configuration d'Azure
    configure_azure_monitor(connection_string=CONNECTION_STRING)
    print("✅ Configuration Azure Monitor : SUCCÈS")

    # 3. Création du logger
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    # 4. Envoi des logs
    print("📤 Envoi d'un log INFO...")
    logger.info("TEST_MANUEL_TITOUAN: Ceci est un test depuis mon PC")
    
    print("📤 Envoi d'un log WARNING...")
    logger.warning("FEEDBACK_USER_ERROR_TEST: Simulation d'erreur pour Azure")
    
    print("⏳ Attente de 10 secondes pour laisser le temps à l'envoi...")
    time.sleep(10)
    print("✅ Fin du script.")

except Exception as e:
    print(f"❌ CRASH: Une erreur s'est produite : {e}")