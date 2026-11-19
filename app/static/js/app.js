(function() {
    'use strict';

    function initTooltips() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    function initPopovers() {
        const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
        popoverTriggerList.map(function(popoverTriggerEl) {
            return new bootstrap.Popover(popoverTriggerEl);
        });
    }

    function showLoading(element) {
        if (!element) return;
        element.innerHTML = `
            <div class="loading-state">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Memuat...</span>
                </div>
                <p class="mb-0">Memuat data...</p>
            </div>
        `;
    }

    function showEmpty(element, message, icon = 'fas fa-inbox') {
        if (!element) return;
        element.innerHTML = `
            <div class="empty-state">
                <i class="${icon}"></i>
                <h5>Tidak Ada Data</h5>
                <p>${message}</p>
            </div>
        `;
    }

    function showError(element, message) {
        if (!element) return;
        element.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle text-danger"></i>
                <h5>Terjadi Kesalahan</h5>
                <p>${message}</p>
                <button class="btn btn-outline-primary btn-sm" onclick="location.reload()">Coba Lagi</button>
            </div>
        `;
    }

    function formatNumber(num, decimals = 2) {
        if (num === null || num === undefined || isNaN(num)) return '-';
        return new Intl.NumberFormat('id-ID', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(num);
    }

    function formatDate(dateString) {
        if (!dateString) return '-';
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return '-';
        return new Intl.DateTimeFormat('id-ID', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        }).format(date);
    }

    function formatDateShort(dateString) {
        if (!dateString) return '-';
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return '-';
        return new Intl.DateTimeFormat('id-ID', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric'
        }).format(date);
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    function apiRequest(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        return fetch(url, { ...defaultOptions, ...options })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => Promise.reject(err));
                }
                return response.json();
            });
    }

    function updateStatusIndicator(status, text) {
        const dot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        if (dot && statusText) {
            dot.className = 'fas fa-circle me-1';
            if (status === 'loading') {
                dot.classList.add('text-warning');
                dot.classList.add('fa-spinner');
                dot.classList.add('fa-spin');
                dot.classList.remove('fa-circle');
            } else if (status === 'success') {
                dot.classList.add('text-success');
            } else if (status === 'error') {
                dot.classList.add('text-danger');
            } else {
                dot.classList.add('text-secondary');
            }
            statusText.textContent = text;
        }
    }

    document.addEventListener('DOMContentLoaded', function() {
        initTooltips();
        initPopovers();

        const darkModeToggle = document.getElementById('dark-mode-toggle');
        if (darkModeToggle) {
            darkModeToggle.addEventListener('change', function() {
                document.documentElement.setAttribute(
                    'data-bs-theme',
                    this.checked ? 'dark' : 'light'
                );
                localStorage.setItem('darkMode', this.checked);
            });
            const savedDarkMode = localStorage.getItem('darkMode') === 'true';
            darkModeToggle.checked = savedDarkMode;
            document.documentElement.setAttribute(
                'data-bs-theme',
                savedDarkMode ? 'dark' : 'light'
            );
        }

        const compactSidebarToggle = document.getElementById('compact-sidebar-toggle');
        if (compactSidebarToggle) {
            compactSidebarToggle.addEventListener('change', function() {
                document.body.classList.toggle('sidebar-mini', this.checked);
                if (this.checked) {
                    document.body.classList.add('sidebar-collapse');
                } else if (window.innerWidth > 991.98) {
                    document.body.classList.remove('sidebar-collapse');
                }
                localStorage.setItem('compactSidebar', this.checked);
            });
            const savedCompact = localStorage.getItem('compactSidebar') === 'true';
            compactSidebarToggle.checked = savedCompact;
            document.body.classList.toggle('sidebar-mini', savedCompact);
            if (savedCompact) {
                document.body.classList.add('sidebar-collapse');
            } else if (window.innerWidth > 991.98) {
                document.body.classList.remove('sidebar-collapse');
            }
        }
    });

    window.EnergiCast = {
        initTooltips,
        initPopovers,
        showLoading,
        showEmpty,
        showError,
        formatNumber,
        formatDate,
        formatDateShort,
        debounce,
        apiRequest,
        updateStatusIndicator
    };
})();