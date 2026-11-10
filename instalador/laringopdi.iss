; Instalador do LaringoPDI para Windows (Inno Setup 6), no mesmo modelo do CadÁgil.
;
; - Instala na pasta de programas da CONTA ATUAL ({autopf} com PrivilegesRequired=lowest), sem pedir senha de administrador.
; - Leva tudo junto: Python 3.11 próprio, PyTorch de CPU (roda em qualquer PC, sem placa de vídeo), os modelos do fold 0,
;   as tabelas do estudo e as imagens de amostra do BAGLS. Não precisa do dataset nem do projeto.
; - O atalho chama o python.exe embutido com o lancador.py, que abre o navegador quando o app está pronto.
;
; Para gerar: instalador\montar_instalador.bat (monta build\pacote e chama o ISCC).

#define Nome "LaringoPDI"
#ifndef Versao
  #define Versao "1.0.0"
#endif

[Setup]
AppId={{6B1E2C7A-3D44-4F0B-9E21-1A9D5C0F7B31}
AppName={#Nome}
AppVersion={#Versao}
AppVerName={#Nome} {#Versao}
AppPublisher=Equipe LaringoPDI · UNIFOR
AppComments=Protótipo do estudo de PDI aplicado à laringoscopia. Apoio acadêmico; não substitui avaliação médica.
VersionInfoVersion={#Versao}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={autopf}\LaringoPDI
DefaultGroupName={#Nome}
DisableProgramGroupPage=yes
DisableDirPage=auto
OutputDir=..\build\instalador
OutputBaseFilename=LaringoPDI-{#Versao}-instalador
SetupIconFile=..\build\pacote\LaringoPDI.ico
UninstallDisplayIcon={app}\LaringoPDI.ico
UninstallDisplayName={#Nome}
Compression=lzma2/normal
SolidCompression=yes
LZMANumBlockThreads=6
LZMAUseSeparateProcess=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "atalhodesktop"; Description: "Criar um atalho na Área de Trabalho"; GroupDescription: "Atalhos:"; Flags: checkedonce

[Files]
Source: "..\build\pacote\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#Nome}"; Filename: "{app}\python\python.exe"; Parameters: """{app}\lancador.py"""; WorkingDir: "{app}"; IconFilename: "{app}\LaringoPDI.ico"; Comment: "Abrir o protótipo do LaringoPDI"
Name: "{autodesktop}\{#Nome}"; Filename: "{app}\python\python.exe"; Parameters: """{app}\lancador.py"""; WorkingDir: "{app}"; IconFilename: "{app}\LaringoPDI.ico"; Comment: "Abrir o protótipo do LaringoPDI"; Tasks: atalhodesktop

[Run]
Filename: "{app}\python\python.exe"; Parameters: """{app}\lancador.py"""; WorkingDir: "{app}"; Description: "Abrir o LaringoPDI agora"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Caches criados durante o uso (__pycache__, pastas de dados vazias).
Type: filesandordirs; Name: "{app}"
