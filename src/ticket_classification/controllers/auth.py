"""Principal authentication, role-based access control, and tenant/domain authorization."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import HTTPException, status

from ticket_classification.config.settings import AppSettings


@dataclass(frozen=True)
class Principal:
    """Authenticated caller identity derived from gateway/API credentials."""

    subject: str
    tenant_id: str
    roles: frozenset[str] = field(default_factory=lambda: frozenset({"reader"}))
    allowed_domains: frozenset[str] = field(default_factory=lambda: frozenset({"*"}))

    def has_role(self, *required: str) -> bool:
        return "admin" in self.roles or bool(self.roles.intersection(required))

    def can_access_domain(self, domain_id: str) -> bool:
        return "*" in self.allowed_domains or domain_id in self.allowed_domains


def resolve_principal(
    settings: AppSettings,
    authorization: str | None = None,
    x_api_key: str | None = None,
) -> Principal:
    """Authenticate caller from Bearer token or X-API-Key when auth_enabled is True."""
    if not settings.auth_enabled:
        return Principal(
            subject="local-dev",
            tenant_id=settings.tenant_id,
            roles=frozenset({"admin", "trainer", "reader"}),
            allowed_domains=frozenset({"*"}),
        )

    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()

    if not token or token not in settings.auth_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authentication credentials",
        )

    entry = settings.auth_tokens[token]
    return Principal(
        subject=str(entry.get("subject", "authenticated-client")),
        tenant_id=str(entry.get("tenant_id", settings.tenant_id)),
        roles=frozenset(entry.get("roles", ["reader"])),
        allowed_domains=frozenset(entry.get("allowed_domains", ["*"])),
    )


def require_roles(principal: Principal, *roles: str) -> None:
    if not principal.has_role(*roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions; requires one of {sorted(roles)}",
        )


def require_domain_access(principal: Principal, domain_id: str) -> None:
    if not principal.can_access_domain(domain_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Principal is not authorized for domain '{domain_id}'",
        )
