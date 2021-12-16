import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

import 'custom'


RowLayout {
    spacing : 10

    property alias file_select : file_select
    property alias model_options : model_options

    CustomBox {
        title : 'Files'

        FileSelect {
            id : file_select
            anchors.fill : parent
        }
    }

    CustomBox {
        title : 'Options'
        padding : 10
        Layout.preferredWidth : 30

        ModelOptions {
            id : model_options
            anchors.fill : parent
        }
    }
}
