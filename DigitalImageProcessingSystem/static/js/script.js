// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    const enterSystemBtn = document.getElementById('enterSystemBtn');
    const mainSubMenu = document.getElementById('mainSubMenu');
    if (enterSystemBtn && mainSubMenu) {
        enterSystemBtn.addEventListener('click', function(event) {
            mainSubMenu.classList.toggle('hidden');
            document.querySelectorAll('.sub-sub-menu:not(.hidden)').forEach(openSubMenu => {
                openSubMenu.classList.add('hidden');
            });
            event.stopPropagation();
        });
    }
    const categoryButtons = document.querySelectorAll('.category-button');
    categoryButtons.forEach(button => {
        button.addEventListener('click', function(event) {
            const targetSubMenuId = this.dataset.targetsubmenu;
            const targetSubMenu = document.getElementById(targetSubMenuId);
            if (targetSubMenu) {
                document.querySelectorAll('.sub-sub-menu:not(.hidden)').forEach(openSubMenu => {
                    if (openSubMenu.id !== targetSubMenuId) { openSubMenu.classList.add('hidden'); }
                });
                targetSubMenu.classList.toggle('hidden');
            }
            event.stopPropagation();
        });
    });
    document.addEventListener('click', function() {
        if (mainSubMenu && !mainSubMenu.classList.contains('hidden')) { mainSubMenu.classList.add('hidden'); }
        document.querySelectorAll('.sub-sub-menu:not(.hidden)').forEach(openSubMenu => { openSubMenu.classList.add('hidden'); });
    });
    if (mainSubMenu) { mainSubMenu.addEventListener('click', function(event) { event.stopPropagation(); }); }
});