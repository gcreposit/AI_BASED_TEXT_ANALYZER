// Multilingual Topic Clustering System - JavaScript Utilities

// Global utilities and common functions
(function() {
    'use strict';

    // Initialize app when DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        initializeApp();
    });

    function initializeApp() {
        initializeTooltips();
        setupGlobalEventListeners();
        checkSystemHealth();
    }

    // Format numbers with commas
    window.formatNumber = function(num) {
        return new Intl.NumberFormat().format(num);
    };

    // Format processing time
    window.formatProcessingTime = function(ms) {
        if (ms < 1000) return `${ms}ms`;
        return `${(ms / 1000).toFixed(1)}s`;
    };

    // Show toast notification
    window.showToast = function(message, type = 'info', duration = 5000) {
        const toastContainer = document.getElementById('toastContainer') || createToastContainer();

        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');

        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-${getToastIcon(type)} me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;

        toastContainer.appendChild(toast);

        const bsToast = new bootstrap.Toast(toast, { delay: duration });
        bsToast.show();

        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    };

    function createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
        return container;
    }

    function getToastIcon(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    // Copy text to clipboard
    window.copyToClipboard = async function(text) {
        try {
            await navigator.clipboard.writeText(text);
            showToast('Copied to clipboard!', 'success');
        } catch (err) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            showToast('Copied to clipboard!', 'success');
        }
    };

    // API call wrapper with error handling
    window.apiCall = async function(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options
        };

        try {
            const response = await fetch(url, defaultOptions);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || data.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            return data;
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    };

    // Debounce function for search inputs
    window.debounce = function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    };

    // Format date for display
    window.formatDate = function(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    // Validate text input
    window.validateTextInput = function(text) {
        if (!text || text.trim().length < 3) {
            throw new Error('Text must be at least 3 characters long');
        }
        if (text.length > 10000) {
            throw new Error('Text must be less than 10,000 characters');
        }
        return true;
    };

    // Initialize tooltips
    function initializeTooltips() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Auto-resize textareas
    window.autoResizeTextarea = function(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    };

    // Setup global event listeners
    function setupGlobalEventListeners() {
        // Auto-resize textareas
        document.addEventListener('input', function(e) {
            if (e.target.tagName === 'TEXTAREA' && e.target.hasAttribute('data-auto-resize')) {
                autoResizeTextarea(e.target);
            }
        });

        // Handle navigation active states
        updateActiveNavigation();

        // Global error handler
        window.addEventListener('error', function(e) {
            console.error('Global error:', e.error);
            showToast('An unexpected error occurred', 'danger');
        });

        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', function(e) {
            console.error('Unhandled promise rejection:', e.reason);
            showToast('A network error occurred', 'warning');
        });
    }

    // Update active navigation item
    function updateActiveNavigation() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    }

    // Check system health periodically
    function checkSystemHealth() {
        const checkHealth = async () => {
            try {
                const health = await apiCall('/health');
                updateHealthIndicator(health.status);
            } catch (error) {
                console.warn('Health check failed:', error);
                updateHealthIndicator('unhealthy');
            }
        };

        // Initial check
        checkHealth();

        // Check every 2 minutes
        setInterval(checkHealth, 120000);
    }

    // Update health indicator in UI
    function updateHealthIndicator(status) {
        let indicator = document.getElementById('healthIndicator');

        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'healthIndicator';
            indicator.className = 'position-fixed bottom-0 end-0 p-2';
            indicator.style.zIndex = '1050';
            document.body.appendChild(indicator);
        }

        const statusConfig = {
            'healthy': { class: 'success', icon: 'check-circle', text: 'System Healthy' },
            'degraded': { class: 'warning', icon: 'exclamation-triangle', text: 'System Degraded' },
            'unhealthy': { class: 'danger', icon: 'times-circle', text: 'System Down' }
        };

        const config = statusConfig[status] || statusConfig['unhealthy'];

        indicator.innerHTML = `
            <div class="badge bg-${config.class} d-flex align-items-center"
                 data-bs-toggle="tooltip"
                 data-bs-placement="left"
                 title="${config.text}">
                <i class="fas fa-${config.icon} me-1"></i>
                <small>${status.charAt(0).toUpperCase() + status.slice(1)}</small>
            </div>
        `;

        // Reinitialize tooltip for new element
        new bootstrap.Tooltip(indicator.querySelector('[data-bs-toggle="tooltip"]'));
    }

    // Common error handler
    window.handleError = function(error, context = '') {
        console.error(`Error in ${context}:`, error);

        let message = 'An error occurred';
        if (error.message) {
            message = error.message;
        } else if (typeof error === 'string') {
            message = error;
        }

        showToast(`Error: ${message}`, 'danger');
    };

    // Performance monitoring
    window.performance_monitor = {
        start: function(label) {
            this[label] = performance.now();
        },

        end: function(label) {
            if (this[label]) {
                const duration = performance.now() - this[label];
                console.log(`⏱️ ${label}: ${duration.toFixed(2)}ms`);
                delete this[label];
                return duration;
            }
        }
    };

    // Loading state management
    window.LoadingManager = {
        show: function(element, text = 'Loading...') {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }

            if (element) {
                element.innerHTML = `
                    <div class="text-center p-4">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">${text}</span>
                        </div>
                        <div class="mt-2 text-muted">${text}</div>
                    </div>
                `;
            }
        },

        hide: function(element) {
            if (typeof element === 'string') {
                element = document.getElementById(element);
            }

            if (element) {
                const spinner = element.querySelector('.spinner-border');
                if (spinner) {
                    spinner.parentElement.remove();
                }
            }
        }
    };

    // Local storage wrapper with error handling
    window.StorageManager = {
        set: function(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (error) {
                console.warn('Failed to save to localStorage:', error);
                return false;
            }
        },

        get: function(key, defaultValue = null) {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (error) {
                console.warn('Failed to read from localStorage:', error);
                return defaultValue;
            }
        },

        remove: function(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (error) {
                console.warn('Failed to remove from localStorage:', error);
                return false;
            }
        }
    };

    // Form validation utilities
    window.FormValidator = {
        validateEmail: function(email) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return emailRegex.test(email);
        },

        validateUrl: function(url) {
            try {
                new URL(url);
                return true;
            } catch {
                return false;
            }
        },

        validateRequired: function(value) {
            return value !== null && value !== undefined && value.toString().trim() !== '';
        },

        validateLength: function(value, min, max) {
            const length = value ? value.toString().length : 0;
            return length >= min && length <= max;
        }
    };

    // Export global functions for use in templates
    window.TopicClusteringApp = {
        showToast,
        copyToClipboard,
        apiCall,
        debounce,
        formatDate,
        formatNumber,
        formatProcessingTime,
        validateTextInput,
        autoResizeTextarea,
        handleError,
        LoadingManager,
        StorageManager,
        FormValidator,
        performance_monitor
    };

})();

// Custom event for app initialization complete
document.addEventListener('DOMContentLoaded', function() {
    const event = new CustomEvent('appInitialized', {
        detail: { timestamp: Date.now() }
    });
    window.dispatchEvent(event);
});

console.log('📱 Multilingual Topic Clustering System - JavaScript loaded successfully');