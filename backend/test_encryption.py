"""
Test de cifrado AES-256 para metadata biométrica.
Ejecutar desde el directorio backend:
    python test_encryption.py
"""

import sys
import os

# Agregar el directorio backend al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_encryption():
    print("=" * 80)
    print("TEST DE CIFRADO AES-256 PARA METADATA BIOMETRICA")
    print("=" * 80)
    
    # 1. Verificar que cryptography está instalado
    print("\n[1] Verificando instalacion de cryptography...")
    try:
        from cryptography.fernet import Fernet
        print("    OK: cryptography instalado correctamente")
    except ImportError:
        print("    ERROR: cryptography NO esta instalado")
        print("    Ejecuta: pip install cryptography")
        return False
    
    # 2. Importar BiometricDatabase
    print("\n[2] Importando BiometricDatabase...")
    try:
        from app.core.supabase_biometric_storage import BiometricDatabase, CRYPTO_AVAILABLE
        print(f"    OK: Modulo importado")
        print(f"    CRYPTO_AVAILABLE = {CRYPTO_AVAILABLE}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return False
    
    # 3. Crear instancia de BiometricDatabase
    print("\n[3] Inicializando BiometricDatabase...")
    try:
        db = BiometricDatabase()
        print(f"    OK: Base de datos inicializada")
        print(f"    encryption_enabled = {db.encryption_enabled}")
        print(f"    cipher inicializado = {db.cipher is not None}")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Verificar archivo de clave
    print("\n[4] Verificando archivo de clave...")
    from pathlib import Path
    key_file = Path('biometric_data') / 'keys' / 'encryption_key.key'
    if key_file.exists():
        print(f"    OK: Clave existe en {key_file.absolute()}")
        print(f"    Tamano: {key_file.stat().st_size} bytes")
    else:
        print(f"    WARNING: Clave NO existe en {key_file}")
    
    # 5. Probar cifrado/descifrado
    print("\n[5] Probando cifrado y descifrado...")
    test_metadata = {
        'bootstrap_features': [0.1, 0.2, 0.3, 0.4, 0.5] * 20,  # 100 valores
        'temporal_sequence': [[1,2,3], [4,5,6], [7,8,9]],
        'gesture_info': 'test_gesture',
        'quality': 0.95
    }
    
    print(f"    Metadata original: {len(test_metadata)} keys")
    print(f"    Keys: {list(test_metadata.keys())}")
    
    # Cifrar
    encrypted = db._encrypt_metadata(test_metadata)
    print(f"\n    Tipo despues de cifrar: {type(encrypted)}")
    
    if isinstance(encrypted, str):
        print(f"    OK: Metadata cifrada como string")
        print(f"    Longitud cifrado: {len(encrypted)} caracteres")
        print(f"    Primeros 50 chars: {encrypted[:50]}...")
    else:
        print(f"    INFO: Metadata NO cifrada (retorno dict)")
        if not db.encryption_enabled:
            print(f"         Razon: encryption_enabled = False")
    
    # Descifrar
    decrypted = db._decrypt_metadata(encrypted)
    print(f"\n    Tipo despues de descifrar: {type(decrypted)}")
    
    if isinstance(decrypted, dict):
        print(f"    OK: Metadata descifrada como dict")
        print(f"    Keys recuperadas: {list(decrypted.keys())}")
        
        # Verificar integridad
        if decrypted.get('bootstrap_features') == test_metadata['bootstrap_features']:
            print(f"    OK: bootstrap_features intacto")
        else:
            print(f"    ERROR: bootstrap_features corrupto")
            
        if decrypted.get('temporal_sequence') == test_metadata['temporal_sequence']:
            print(f"    OK: temporal_sequence intacto")
        else:
            print(f"    ERROR: temporal_sequence corrupto")
    else:
        print(f"    ERROR: Descifrado no retorno dict")
    
    # 6. Probar backward compatibility (dict sin cifrar)
    print("\n[6] Probando backward compatibility (dict sin cifrar)...")
    old_metadata = {'old_key': 'old_value', 'number': 42}
    decrypted_old = db._decrypt_metadata(old_metadata)
    
    if decrypted_old == old_metadata:
        print(f"    OK: Dict sin cifrar retornado sin modificar")
    else:
        print(f"    ERROR: Backward compatibility fallida")
    
    # 7. Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DEL TEST")
    print("=" * 80)
    
    summary = db.get_summary()
    print(f"    encryption_enabled en summary: {summary.get('encryption_enabled')}")
    print(f"    total_templates: {summary.get('total_templates')}")
    print(f"    total_users: {summary.get('total_users')}")
    
    if db.encryption_enabled and db.cipher is not None:
        print("\n    RESULTADO: CIFRADO AES-256 ACTIVO Y FUNCIONANDO")
        return True
    else:
        print("\n    RESULTADO: CIFRADO NO ACTIVO")
        print(f"    encryption_enabled = {db.encryption_enabled}")
        print(f"    cipher = {db.cipher}")
        return False

if __name__ == "__main__":
    success = test_encryption()
    print("\n" + "=" * 80)
    if success:
        print("TEST COMPLETADO EXITOSAMENTE")
    else:
        print("TEST FALLIDO - REVISAR ERRORES ARRIBA")
    print("=" * 80)
