// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    const enterSystemBtn = document.getElementById('enterSystemBtn');
    const mainSubMenu = document.getElementById('mainSubMenu');

    if (enterSystemBtn && mainSubMenu) {
        enterSystemBtn.addEventListener('click', function(event) {
            mainSubMenu.classList.toggle('hidden');
            // 关闭其他可能打开的类别子菜单
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
                // 点击一个类别按钮时，先关闭所有其他打开的类别子菜单
                document.querySelectorAll('.sub-sub-menu:not(.hidden)').forEach(openSubMenu => {
                    if (openSubMenu.id !== targetSubMenuId) {
                        openSubMenu.classList.add('hidden');
                    }
                });
                // 然后切换当前点击的类别子菜单
                targetSubMenu.classList.toggle('hidden');
            }
            event.stopPropagation();
        });
    });

    // 点击页面其他地方时，隐藏所有打开的菜单
    document.addEventListener('click', function() {
        if (mainSubMenu && !mainSubMenu.classList.contains('hidden')) {
            mainSubMenu.classList.add('hidden');
        }
        document.querySelectorAll('.sub-sub-menu:not(.hidden)').forEach(openSubMenu => {
            openSubMenu.classList.add('hidden');
        });
    });

    // 阻止主子菜单内部的点击事件冒泡，导致菜单关闭
    if (mainSubMenu) {
        mainSubMenu.addEventListener('click', function(event) {
            event.stopPropagation();
        });
    }
});