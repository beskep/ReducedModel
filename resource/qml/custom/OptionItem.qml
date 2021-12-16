import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15


Item {
    property alias label : label
    property alias text_field : text_field
    property alias unit : unit
    property alias value : text_field.text
    property alias validator : text_field.validator
    property var option_id: 'default'

    Layout.preferredHeight : 45
    Layout.fillWidth : true
    Layout.fillHeight : true

    Component.onCompleted : option()

    function option() {
        con.set_option(option_id + '|' + text_field.text);
    }

    RowLayout {
        anchors.fill : parent

        Label {
            id : label
            Layout.fillWidth : true
            Layout.preferredWidth : 20
            text : ''
        }

        TextField {
            id : text_field
            Layout.fillWidth : true
            Layout.preferredWidth : 10
            selectByMouse : true
            text : ''

            validator : DoubleValidator {}

            onEditingFinished : option()
        }

        Label {
            id : unit
            Layout.fillWidth : true
            Layout.preferredWidth : 5
            text : ''
        }
    }
}
