function runTaskList() {
    var tasklist = document.getElementsByClassName('awaiting-user-config');
    for (var i = 0; i < tasklist.length; i++)
        tasklist[i].children[2].click();
}
function confirmAll() {
    var ok = document.getElementsByClassName('goog-buttonset-default goog-buttonset-action');
    for (var i = 0; i < ok.length; i++)
        ok[i].click();
}
runTaskList();
confirmAll();