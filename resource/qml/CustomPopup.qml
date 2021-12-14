import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15


Popup {
    id : popup

    property alias title : label_title.text
    property alias message : label_message.text

    anchors.centerIn : Overlay.overlay

    padding : 20
    width : 300
    height : 200

    ColumnLayout {
        anchors.fill : parent
        id : column_layout
        spacing : 10

        RowLayout {
            Text {
                id : _icon
                text : '\ue88e'
                font.family : 'Material Icons Outlined'
                font.pointSize : 20
            }

            Label {
                id : label_title
                text : 'title'
                font.pointSize : 14
                font.weight : Font.Medium
                Layout.fillWidth : true
                Layout.alignment : Qt.AlignVCenter
            }
        }

        Label {
            id : label_message
            text : 'message'
            font.pointSize : 12
            Layout.fillWidth : true
            wrapMode : Label.WordWrap
        }

        Button {
            text : 'OK'
            Layout.alignment : Qt.AlignRight | Qt.AlignBottom
            flat : true

            onClicked : {
                popup.close()
            }
        }
    }

    function set_icon(level = 0) {
        if (level === 0) {
            _icon.text = '\ue5ca' // check
        } else if (level === 1) {
            _icon.text = '\ue88e' // info
        } else if (level === 2) {
            _icon.text = '\uf083' // warning
        }
    }
}
