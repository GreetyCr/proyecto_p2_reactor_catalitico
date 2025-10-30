# 📤 Guía: Subir Proyecto a GitHub

## ✅ Estado Actual

- ✅ Commit realizado localmente
- ✅ README.md completo y profesional
- ✅ LICENSE (MIT) añadido
- ✅ .gitignore configurado correctamente
- ✅ 32 archivos en el commit

---

## 🚀 Pasos para Crear el Repositorio en GitHub

### 1. Crear el Repositorio en GitHub

1. Ve a [https://github.com/new](https://github.com/new)
2. Configura el repositorio:
   - **Repository name**: `proyecto_p2_reactor_catalitico`
   - **Description**: `Simulación 2D de transferencia de masa en reactor catalítico con Crank-Nicolson`
   - **Visibilidad**: 🔓 Public (o 🔒 Private si prefieres)
   - ⚠️ **NO marques**: "Initialize this repository with a README" (ya tenemos uno)
   - ⚠️ **NO agregues**: .gitignore ni licencia (ya los tenemos)
3. Click en **"Create repository"**

---

### 2. Conectar tu Repositorio Local con GitHub

GitHub te mostrará instrucciones, pero estos son los comandos exactos para tu caso:

```bash
cd /Users/randallbonilla/Desktop/proyecto_p2_reactor_catalitico

# Agregar el remote de GitHub (reemplaza TU-USUARIO con tu username de GitHub)
git remote add origin https://github.com/TU-USUARIO/proyecto_p2_reactor_catalitico.git

# Verificar que se agregó correctamente
git remote -v

# Hacer push del commit al repositorio
git push -u origin main
```

---

### 3. Verificar la Subida

Después del `git push`:

1. Ve a tu repositorio en GitHub: `https://github.com/TU-USUARIO/proyecto_p2_reactor_catalitico`
2. Deberías ver:
   - ✅ README.md bien formateado con badges
   - ✅ Estructura de carpetas completa
   - ✅ 32 archivos subidos
   - ✅ Licencia MIT

---

## 🔐 Autenticación

Si GitHub te pide credenciales:

### Opción A: HTTPS con Token (Recomendado)
1. Ve a GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Genera un token con permisos de `repo`
3. Usa el token como contraseña cuando git te lo pida

### Opción B: SSH (Más conveniente a largo plazo)
```bash
# Generar clave SSH si no tienes una
ssh-keygen -t ed25519 -C "tu-email@ejemplo.com"

# Copiar la clave pública
cat ~/.ssh/id_ed25519.pub

# Agregar la clave en GitHub: Settings → SSH and GPG keys → New SSH key
# Cambiar el remote a SSH
git remote set-url origin git@github.com:TU-USUARIO/proyecto_p2_reactor_catalitico.git
```

---

## 📊 Próximos Pasos (Opcional)

### Habilitar GitHub Pages (si quieres documentación web)
1. En tu repo GitHub: Settings → Pages
2. Source: `Deploy from a branch`
3. Branch: `main` → folder: `/docs` (si tienes Sphinx docs)
4. Save

### Agregar Badges al README
Ya incluí algunos badges básicos, pero puedes agregar más:
- Build status (si usas CI/CD)
- DOI (si publicas en Zenodo)
- Colab notebook (si haces uno interactivo)

---

## 🐛 Troubleshooting

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/TU-USUARIO/proyecto_p2_reactor_catalitico.git
```

### Error: "failed to push some refs"
```bash
git pull origin main --rebase
git push -u origin main
```

### Quiero cambiar el nombre del repositorio después
1. GitHub → Settings → Repository name
2. Localmente: `git remote set-url origin <nueva-URL>`

---

## ✅ Checklist Final

Antes de compartir el repositorio:

- [ ] Push exitoso a GitHub
- [ ] README se ve bien en GitHub (con imágenes)
- [ ] Licencia visible en el repo
- [ ] `.gitignore` funcionando (no se subieron `venv/` ni `data/output/`)
- [ ] Estructura de carpetas correcta
- [ ] Tests documentados en README
- [ ] Badges funcionando

---

## 🎉 ¡Listo!

Una vez que hagas push, tu proyecto estará en GitHub y podrás:
- 📤 Compartir el link con tu profesor
- 🌟 Recibir estrellas de la comunidad
- 📝 Colaborar con otros (si es público)
- 💾 Tener respaldo en la nube

---

**Tu URL final será**: `https://github.com/TU-USUARIO/proyecto_p2_reactor_catalitico`

¡Mucho éxito! 🚀

