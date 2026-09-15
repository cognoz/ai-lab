terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
  # Reuse your existing remote-state backend pattern (azurerm backend).
  # Fill in to match your shared/cluster state naming.
  backend "azurerm" {
    resource_group_name  = "rg-tfstate-devsecops-lab"
    storage_account_name = "tfstatedevsecopslabrk"
    container_name       = "tfstate"
    key                  = "ai-lab-pg.tfstate"
  }
}

provider "azurerm" {
  features {}
}

variable "location" {
  type    = string
  default = "swedencentral"
}

variable "resource_group_name" {
  type    = string
  default = "rg-ai-lab"
}

variable "pg_admin_login" {
  type    = string
  default = "pgadmin"
}

variable "pg_admin_password" {
  type      = string
  sensitive = true
}

variable "home_ip" {
  type        = string
  description = "Your home/public IP for the firewall allow rule"
}

variable "office_ip" {
  type        = string
  description = "Your office/public IP for the firewall allow rule"
}

resource "azurerm_resource_group" "ai_lab" {
  name     = var.resource_group_name
  location = var.location
}

resource "azurerm_postgresql_flexible_server" "pg" {
  name                          = "pg-ai-lab-rk"
  resource_group_name           = azurerm_resource_group.ai_lab.name
  location                      = azurerm_resource_group.ai_lab.location
  version                       = "16"
  administrator_login           = var.pg_admin_login
  administrator_password        = var.pg_admin_password
  public_network_access_enabled = true

  # Burstable tier — cheapest, plenty for Phase 1.
  sku_name   = "B_Standard_B1ms"
  storage_mb = 32768

  # No HA / long backups for a learning lab; keep the bill minimal.
  backup_retention_days = 7
  zone                  = "1"
}

resource "azurerm_postgresql_flexible_server_configuration" "extensions" {
  name      = "azure.extensions"
  server_id = azurerm_postgresql_flexible_server.pg.id
  value     = "VECTOR"
}

resource "azurerm_postgresql_flexible_server_database" "ragdb" {
  name      = "ragdb"
  server_id = azurerm_postgresql_flexible_server.pg.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Firewall: home
resource "azurerm_postgresql_flexible_server_firewall_rule" "home" {
  name             = "allow-home-ip"
  server_id        = azurerm_postgresql_flexible_server.pg.id
  start_ip_address = var.home_ip
  end_ip_address   = var.home_ip
}

# Firewall: office
resource "azurerm_postgresql_flexible_server_firewall_rule" "office" {
  name             = "allow-office-ip"
  server_id        = azurerm_postgresql_flexible_server.pg.id
  start_ip_address = var.office_ip
  end_ip_address   = var.office_ip
}

output "database_url" {
  value = format(
    "postgres://%s:%s@%s/ragdb?sslmode=require",
    var.pg_admin_login,
    var.pg_admin_password,
    azurerm_postgresql_flexible_server.pg.fqdn,
  )
  sensitive = true
}

