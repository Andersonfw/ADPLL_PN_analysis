#ssh andersonfw@191.4.204.148:/home/andersonfw/Digital/ADPLL_WIMED/sim/data/* data/
# Script para baixar arquivos de um servidor remoto via SCP
#Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

Write-Host "Iniciando a transferência de arquivos..."

# Define as variáveis (opcional, mas recomendado)
$usuarioRemoto = "andersonfw"
$servidorRemoto = "191.4.204.148"
$caminhoRemoto = "/home/andersonfw/Digital/ADPLL_WIMED/sim/data/*"
$diretorioLocal = ".\data" # O ponto significa o diretório atual

# Cria o diretório local se ele não existir
if (-not (Test-Path -Path $diretorioLocal)) {
    Write-Host "Criando o diretório local: $diretorioLocal"
    New-Item -ItemType Directory -Path $diretorioLocal
}

# Monta e executa o comando SCP
Write-Host "Executando o comando SCP para baixar os arquivos..."
scp "$($usuarioRemoto)@$($servidorRemoto):$($caminhoRemoto)" $diretorioLocal

Write-Host "Transferencia de arquivos concluida!"