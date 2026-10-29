========================================================================
TINYDERM VISION — CLIENT-SIDE IN-BROWSER WEB SCANNER DEPLOYMENT
========================================================================

This package runs 100% inside the client's web browser using ONNX Runtime Web.
No Python, no backend server, and no cloud GPU are needed on your web host!

HOW TO DEPLOY ON YOUR WEBSITE:
------------------------------
1. Unzip the contents of this package into your website's folder or subfolder.
   Example destinations:
   - Root: https://yourdomain.com/
   - Subfolder: https://yourdomain.com/tinyderm/

2. Ensure your server hosts:
   - index.html
   - model.onnx (must be in the same directory as index.html)

3. IMPORTANT BROWSER CAMERA SECURITY REQUIREMENT:
   Web browsers (Chrome, Safari, iOS, Android, Firefox, Edge) ONLY permit camera
   access over HTTPS or on http://localhost.
   Make sure your website has SSL/HTTPS enabled (https://yourdomain.com).

SUPPORTED HOSTING PROVIDERS (ANY STATIC HOST):
----------------------------------------------
- Apache / Nginx / cPanel / Shared Hosting
- GitHub Pages
- Cloudflare Pages
- Netlify / Vercel
- AWS S3 / CloudFront / Firebase Hosting

FILES INCLUDED:
---------------
- index.html: Complete HTML5/CSS/JavaScript client-side web application with
  camera viewfinder, target reticle, auto-capture, synthesized Web Audio "Ding!"
  chime, confidence meter, and research medical disclaimer.
- model.onnx: MobileNetV3-Small deployable ONNX model trained on ISIC 2018 / HAM10000.
